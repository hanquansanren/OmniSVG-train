import numpy as np
import torch
import yaml
from typing import List, Tuple, Dict, Optional, Union
from deepsvg.difflib.tensor import SVGTensor
from deepsvg.svglib.svg import SVG
from deepsvg.svglib.geom import Bbox

try:
    from utils.config import TokenizationConfig
except ImportError:
    TokenizationConfig = None


class SVGTokenizer:
    """SVG tokenizer - supports both 8B and 4B models via config.yaml"""
    
    def __init__(self, config_path: str = "./config.yaml", model_size: str = None):
        """
        Initialize SVGTokenizer.
        
        Args:
            config_path: Path to config.yaml
            model_size: Model size ("8B" or "4B"). If None, uses default from config.
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Determine model size
        self.model_size = model_size or self.config.get('default_model_size', '8B')
        if self.model_size not in self.config.get('models', {}):
            raise ValueError(f"Invalid model_size: {self.model_size}. Must be one of: {list(self.config.get('models', {}).keys())}")
        
        self._load_config()
        self.pixel2xy = self._create_pixel2xy_mapping()
    
    def _get_model_specific_config(self, *keys):
        """Get model-specific config value, with fallback to shared config."""
        model_cfg = self.config.get('models', {}).get(self.model_size, {})
        
        # Navigate through nested keys in model-specific config
        value = model_cfg
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                value = None
                break
        
        # If not found in model-specific, try shared config
        if value is None:
            value = self.config
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return None
        
        return value
    
    def _load_config(self):
        """Load all constants from configuration file with model-specific overrides."""
        # ========== Token-related configs ==========
        # Model-specific tokens
        self.NUM_MASK_AND_EOM = self._get_model_specific_config('tokens', 'num_mask_and_eom')
        self.BASE_OFFSET = self._get_model_specific_config('tokens', 'base_offset')
        
        # Shared tokens
        tokens_cfg = self.config['tokens']
        self.NUM_SVG_END = tokens_cfg['svg_end']
        self.NUM_END_TOKEN = tokens_cfg['num_end_token']
        
        # ========== Coordinate-related configs ==========
        # Model-specific coordinates
        self.PIX_PAD = self._get_model_specific_config('coordinates', 'pix_pad_offset')
        self.COORD_PAD = self._get_model_specific_config('coordinates', 'coord_pad_offset')
        
        # Shared coordinates
        coords_cfg = self.config['coordinates']
        self.BBOX = coords_cfg['bbox']
        
        # ========== Color-related configs ==========
        colors_cfg = self.config['colors']
        self.COLOR_TOKEN_START_RAW = colors_cfg['color_token_start']
        self.MAX_COLOR_TOKENS = colors_cfg['max_color_tokens']
        
        # Model-specific colors
        self.COLOR_START_OFFSET = self._get_model_specific_config('colors', 'color_start_offset')
        self.COLOR_END_OFFSET = self._get_model_specific_config('colors', 'color_end_offset')
        
        # ========== SVG command values ==========
        commands_cfg = self.config['svg_commands']
        self.CMD_MOVE = commands_cfg['move']
        self.CMD_LINE = commands_cfg['line']
        self.CMD_CURVE = commands_cfg['curve']
        self.CMD_ARC = commands_cfg['arc']
        self.CMD_CLOSE = commands_cfg['close']
        
        # ========== Model-related configs ==========
        model_cfg = self.config['model']
        self.BOS_TOKEN_ID = model_cfg['bos_token_id']
        self.EOS_TOKEN_ID = model_cfg['eos_token_id']
        self.PAD_TOKEN_ID = model_cfg['pad_token_id']
        
        # ========== Arc parameter configs ==========
        arc_cfg = self.config.get('arc', {})
        self.ARC_PARAM_OFFSET = arc_cfg.get('param_offset', 44500)
        self.ARC_PARAM_RANGE = arc_cfg.get('param_range', 100)
        self.ARC_PARAM_START = self.ARC_PARAM_OFFSET + self.BASE_OFFSET
        
        # ========== Derived constants ==========
        self.PIXEL_OFFSET = (self.NUM_MASK_AND_EOM - self.BASE_OFFSET + 
                             self.NUM_SVG_END - self.CMD_MOVE)
        
        # Command token range
        self.CMD_TOKEN_START = self.NUM_MASK_AND_EOM + self.NUM_SVG_END
        self.CMD_TOKEN_END = self.PIX_PAD + self.NUM_SVG_END
        
        # Coordinate token start
        self.COORD_TOKEN_START = self.PIX_PAD + self.NUM_SVG_END
        
        # Color-coordinate boundary
        self.COLOR_COORD_BOUNDARY = self.COLOR_TOKEN_START_RAW + 1 + self.BASE_OFFSET
        
        # Color threshold for raster_svg
        self.COLOR_THRESHOLD = self.COLOR_TOKEN_START_RAW - self.PIXEL_OFFSET + 1
        
    def _create_pixel2xy_mapping(self) -> Dict[int, np.ndarray]:
        """Create pixel to xy mapping following dataset.py logic."""
        pixel2xy = {}
        x = np.linspace(0, self.BBOX - 1, self.BBOX)
        y = np.linspace(0, self.BBOX - 1, self.BBOX)
        xx, yy = np.meshgrid(x, y)
        xy_grid = (np.array((xx.ravel(), yy.ravel())).T).astype(int)
        
        for pixel, xy in enumerate(xy_grid):
            pixel2xy[pixel] = xy + self.COORD_PAD + self.NUM_SVG_END
            
        return pixel2xy
    
    def token_to_color(self, color_token: int) -> str:
        """Convert token to color following dataset.py logic."""
        try:
            if color_token == self.COLOR_TOKEN_START_RAW:
                return "none"
            elif color_token == self.COLOR_TOKEN_START_RAW + 1:
                return "currentColor"
            
            color_index = color_token - (self.COLOR_TOKEN_START_RAW + 2)
            
            if color_index < 0 or color_index >= self.MAX_COLOR_TOKENS:
                print(f"Warning: Color token {color_token} out of range")
                return "#808080"
            
            r = (color_index >> 8) & 0xF
            g = (color_index >> 4) & 0xF
            b = color_index & 0xF
            
            r = (r << 4) | r
            g = (g << 4) | g
            b = (b << 4) | b
            
            return f"#{r:02x}{g:02x}{b:02x}"
            
        except Exception as e:
            print(f"Error in token_to_color: {e}")
            return "#808080"

    def process_generated_tokens(self, output_ids: torch.Tensor) -> np.ndarray:
        """Process generated tokens following dataset.py logic."""
        # Remove bos/eos
        generated_pixels = output_ids[:, 1:-1].cpu().numpy().flatten() # 通过索引去掉bos和eos
        
        sample_xys = []
        
        for pixel in generated_pixels:
            try: # 根据token ID分类，转换为xy坐标
                # 1. Command tokens: CMD_TOKEN_START <= pixel < CMD_TOKEN_END
                if self.CMD_TOKEN_START <= pixel < self.CMD_TOKEN_END: # 151939 <= pixel < 151944 
                    xy = np.array([pixel - self.BASE_OFFSET, 
                                   pixel - self.BASE_OFFSET]).astype(int)
                    sample_xys.append(xy)
                    
                # 2. Coordinate tokens: COORD_TOKEN_START <= pixel < COLOR_COORD_BOUNDARY
                elif self.COORD_TOKEN_START <= pixel < self.COLOR_COORD_BOUNDARY: # 151944 <= pixel < 191947
                    pixel_index = pixel - self.COORD_TOKEN_START
                    if pixel_index in self.pixel2xy:
                        xy = self.pixel2xy[pixel_index] - self.BASE_OFFSET
                        sample_xys.append(xy)
                    
                # 3. Arc parameters: ARC_PARAM_START + 1 <= pixel < ARC_PARAM_START + 1 + ARC_PARAM_RANGE
                elif (self.ARC_PARAM_START + 1 <= pixel <  # 196437 <= pixel < 196537
                      self.ARC_PARAM_START + 1 + self.ARC_PARAM_RANGE):
                    value = pixel - self.ARC_PARAM_START - 1
                    xy = np.array([value, value]).astype(int)
                    sample_xys.append(xy)
                    
                # 4. Color tokens: COLOR_COORD_BOUNDARY <= pixel < ARC_PARAM_START
                elif self.COLOR_COORD_BOUNDARY <= pixel < self.ARC_PARAM_START:
                    xy = np.array([pixel - self.BASE_OFFSET, 
                                   pixel - self.BASE_OFFSET]).astype(int)
                    sample_xys.append(xy)
                    
            except Exception as e:
                print(f"Error processing pixel {pixel}: {e}")
                continue
        
        if sample_xys:
            return np.vstack(sample_xys) # (2042, 2)
        else:
            return np.array([]).reshape(0, 2)

    def raster_svg(self, pixels: np.ndarray) -> Tuple[List[List[torch.Tensor]], List[int]]:
        """Convert pixels to SVG tensors following dataset.py logic."""
        try:
            if len(pixels) == 0:
                return [[]], []
            
            # Key step: subtract PIXEL_OFFSET
            pixels = pixels - self.PIXEL_OFFSET
            
            svg_tensors = []
            color_tensors = []
            path_tensor = []
            
            i = 0
            while i < len(pixels):
                try:
                    pix = pixels[i]
                    
                    # Move command
                    if pix[0] == self.CMD_MOVE:
                        if i + 2 >= len(pixels):
                            break
                        cmd_tensor = np.zeros(14)
                        cmd_tensor[0] = 0  # Move command index
                        cmd_tensor[12:14] = pixels[i+2]
                        path_tensor.append(cmd_tensor.tolist())
                        i += 3
                        
                    # Line command
                    elif pix[0] == self.CMD_LINE:
                        if i + 1 >= len(pixels):
                            break
                        cmd_tensor = np.zeros(14)
                        cmd_tensor[0] = 1  # Line command index
                        cmd_tensor[12:14] = pixels[i+1]
                        path_tensor.append(cmd_tensor.tolist())
                        i += 2
                        
                    # Curve command
                    elif pix[0] == self.CMD_CURVE:
                        if i + 3 >= len(pixels):
                            break
                        cmd_tensor = np.zeros(14)
                        cmd_tensor[0] = 2  # Curve command index
                        cmd_tensor[8:10] = pixels[i+1]
                        cmd_tensor[10:12] = pixels[i+2]
                        cmd_tensor[12:14] = pixels[i+3]
                        path_tensor.append(cmd_tensor.tolist())
                        i += 4
                        
                    # Arc command
                    elif pix[0] == self.CMD_ARC:
                        if i + 5 >= len(pixels):
                            break
                        cmd_tensor = np.zeros(14)
                        cmd_tensor[0] = 3  # Arc command index
                        radius = pixels[i+1]
                        x_axis_rot = pixels[i+2][0] + self.PIXEL_OFFSET
                        large_arc_flg = pixels[i+3][0] + self.PIXEL_OFFSET
                        sweep_flg = pixels[i+4][0] + self.PIXEL_OFFSET
                        end_pos = pixels[i+5]
                        cmd_tensor[1:3] = radius
                        cmd_tensor[3] = x_axis_rot
                        cmd_tensor[4] = large_arc_flg
                        cmd_tensor[5] = sweep_flg
                        cmd_tensor[12:14] = end_pos
                        path_tensor.append(cmd_tensor.tolist())
                        i += 6
                        
                    # Close command
                    elif pix[0] == self.CMD_CLOSE:
                        if i + 1 >= len(pixels):
                            break
                        cmd_tensor = np.zeros(14)
                        cmd_tensor[0] = 6  # Close command index
                        cmd_tensor[12:14] = pixels[i+1]
                        path_tensor.append(cmd_tensor.tolist())
                        i += 2
                        
                    # Color token: pix[0] >= COLOR_THRESHOLD
                    elif pix[0] >= self.COLOR_THRESHOLD:# 40003
                        if path_tensor:
                            svg_tensors.append(torch.tensor(path_tensor))
                            # Reverse transform: restore original color token
                            color_token = int(pix[0] + self.PIXEL_OFFSET - 1)
                            color_tensors.append(color_token)
                            path_tensor = []
                        i += 1
                    else:
                        i += 1
                        
                except (IndexError, TypeError) as e:
                    print(f"Error at position {i}: {e}")
                    break
            
            # Handle remaining path (without color)
            if path_tensor:
                svg_tensors.append(torch.tensor(path_tensor))
                
            return [svg_tensors], color_tensors
            
        except Exception as e:
            print(f"Error in raster_svg: {e}")
            import traceback
            traceback.print_exc()
            return [[]], []

    def apply_colors_to_svg(self, svg_tensors: List[torch.Tensor], 
                           colors: Optional[List[int]]) -> SVG:
        """Apply colors and create final SVG."""
        paths = []
        
        if not svg_tensors:
            raise ValueError("No valid SVG tensors")
        
        colors = colors or []
        
        for i, path_tensor in enumerate(svg_tensors):
            try:
                path = SVGTensor.from_data(path_tensor)
                path = SVG.from_tensor(path.data, viewbox=Bbox(self.BBOX))
                
                actual_color = self.token_to_color(colors[i]) if i < len(colors) else "none"
                
                for path_group in path:
                    path_group.color = actual_color
                    path_group.stroke_color = "none"
                    
                path.fill_(True)
                paths.append(path)
                
            except Exception as e:
                print(f"Error processing path {i}: {e}")
                continue
        
        if not paths:
            raise ValueError("No valid paths generated")
        
        path_groups = paths[0].svg_path_groups
        for i in range(1, len(paths)):
            path_groups.extend(paths[i].svg_path_groups)
        
        return SVG(path_groups, viewbox=Bbox(self.BBOX))


class TrainAlignedSVGTokenizer:
    """SVG tokenizer aligned with training-time TokenizationConfig.

    Decodes generated token IDs using absolute token ranges that exactly
    match the encoding in utils/dataset.py SVGTokenizer.  No PIXEL_OFFSET
    indirection — every value is decoded to its final form in one step.
    # 用于兼容旧配置训练的模型，在推理时加 --use-train-tokenizer时触发
    """

    # Sentinel command indices used in the intermediate xy-pair representation.
    # These are negative so they never collide with coordinate values (0-199)
    # or color-offset values (40010+).
    _CMD_MOVE = -5
    _CMD_LINE = -4
    _CMD_CURVE = -3
    _CMD_ARC = -2
    _CMD_CLOSE = -1

    def __init__(self, token_config: "TokenizationConfig"):
        cfg = token_config
        self.cfg = cfg

        # Absolute token ranges (must match training encoder in dataset.py)
        self.CMD_TOKENS = {
            cfg.cmd_move:  self._CMD_MOVE,
            cfg.cmd_line:  self._CMD_LINE,
            cfg.cmd_curve: self._CMD_CURVE,
            cfg.cmd_arc:   self._CMD_ARC,
            cfg.cmd_close: self._CMD_CLOSE,
        }
        self.CMD_RANGE = (cfg.cmd_move, cfg.cmd_close + 1)

        self.COORD_START = cfg.pix_pad_offset
        self.COORD_END = cfg.pix_pad_offset + cfg.bbox_size ** 2

        # Color tokens are encoded with +num_svg_end in training
        self.COLOR_ABS_START = cfg.color_token_offset + cfg.base_vocab_size + cfg.num_svg_end
        self.COLOR_ABS_END = cfg.arc_param_start

        self.ARC_START = cfg.arc_param_start
        self.ARC_RANGE = 100

        self.BBOX = cfg.bbox_size
        self.COLOR_TOKEN_OFFSET = cfg.color_token_offset
        self.MAX_COLOR_TOKENS = cfg.max_color_tokens

        self.BOS_TOKEN_ID = cfg.bos_token_id
        self.EOS_TOKEN_ID = cfg.eos_token_id

        self._build_coord_lookup()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _build_coord_lookup(self):
        """Build token -> (x, y) coordinate lookup table."""
        self._token_to_xy: Dict[int, np.ndarray] = {}
        for y in range(self.BBOX):
            for x in range(self.BBOX):
                pixel_index = x + y * self.BBOX
                token = pixel_index + self.COORD_START
                self._token_to_xy[token] = np.array([x, y], dtype=int)

    # ------------------------------------------------------------------
    # Public API  (same interface as SVGTokenizer)
    # ------------------------------------------------------------------

    def process_generated_tokens(self, output_ids: torch.Tensor) -> np.ndarray:
        """Classify each generated token and convert to an xy-pair.

        Encoding in the returned Nx2 array:
          commands   -> [sentinel, sentinel]   (negative, see _CMD_*)
          coords     -> [x, y]                 (0 .. BBOX-1)
          colors     -> [raw_offset, raw_offset] (COLOR_TOKEN_OFFSET+)
          arc params -> [value, value]          (0 .. ARC_RANGE-1)
        """
        tokens = output_ids[:, 1:-1].cpu().numpy().flatten()

        sample_xys: List[np.ndarray] = []
        cmd_lo, cmd_hi = self.CMD_RANGE

        for tok in tokens:
            tok = int(tok)

            if cmd_lo <= tok < cmd_hi:
                idx = self.CMD_TOKENS[tok]
                sample_xys.append(np.array([idx, idx], dtype=int))

            elif self.COORD_START <= tok < self.COORD_END:
                xy = self._token_to_xy.get(tok)
                if xy is not None:
                    sample_xys.append(xy)

            elif self.COLOR_ABS_START <= tok < self.COLOR_ABS_END:
                raw = tok - self.cfg.base_vocab_size - self.cfg.num_svg_end
                sample_xys.append(np.array([raw, raw], dtype=int))

            elif self.ARC_START <= tok < self.ARC_START + self.ARC_RANGE:
                val = tok - self.ARC_START
                sample_xys.append(np.array([val, val], dtype=int))

        if sample_xys:
            return np.vstack(sample_xys)
        return np.array([], dtype=int).reshape(0, 2)

    def raster_svg(
        self, pixels: np.ndarray
    ) -> Tuple[List[List[torch.Tensor]], List[int]]:
        """Build SVG path tensors and collect color tokens from xy-pairs."""
        try:
            if len(pixels) == 0:
                return [[]], []

            svg_tensors: List[torch.Tensor] = []
            color_tensors: List[int] = []
            path_tensor: List[list] = []

            i = 0
            while i < len(pixels):
                try:
                    v = pixels[i][0]

                    if v == self._CMD_MOVE:
                        if i + 2 >= len(pixels):
                            break
                        cmd = np.zeros(14)
                        cmd[0] = 0
                        cmd[12:14] = pixels[i + 2]
                        path_tensor.append(cmd.tolist())
                        i += 3

                    elif v == self._CMD_LINE:
                        if i + 1 >= len(pixels):
                            break
                        cmd = np.zeros(14)
                        cmd[0] = 1
                        cmd[12:14] = pixels[i + 1]
                        path_tensor.append(cmd.tolist())
                        i += 2

                    elif v == self._CMD_CURVE:
                        if i + 3 >= len(pixels):
                            break
                        cmd = np.zeros(14)
                        cmd[0] = 2
                        cmd[8:10] = pixels[i + 1]
                        cmd[10:12] = pixels[i + 2]
                        cmd[12:14] = pixels[i + 3]
                        path_tensor.append(cmd.tolist())
                        i += 4

                    elif v == self._CMD_ARC:
                        if i + 5 >= len(pixels):
                            break
                        cmd = np.zeros(14)
                        cmd[0] = 3
                        cmd[1:3] = pixels[i + 1]       # radius (x, y)
                        cmd[3] = pixels[i + 2][0]       # x_axis_rotation
                        cmd[4] = pixels[i + 3][0]       # large_arc_flag
                        cmd[5] = pixels[i + 4][0]       # sweep_flag
                        cmd[12:14] = pixels[i + 5]      # end position
                        path_tensor.append(cmd.tolist())
                        i += 6

                    elif v == self._CMD_CLOSE:
                        if i + 1 >= len(pixels):
                            break
                        cmd = np.zeros(14)
                        cmd[0] = 6
                        cmd[12:14] = pixels[i + 1]
                        path_tensor.append(cmd.tolist())
                        i += 2

                    elif v >= self.COLOR_TOKEN_OFFSET:
                        if path_tensor:
                            svg_tensors.append(torch.tensor(path_tensor))
                            color_tensors.append(int(v))
                            path_tensor = []
                        i += 1
                    else:
                        i += 1

                except (IndexError, TypeError) as exc:
                    print(f"TrainAlignedSVGTokenizer.raster_svg error at {i}: {exc}")
                    break

            if path_tensor:
                svg_tensors.append(torch.tensor(path_tensor))

            return [svg_tensors], color_tensors

        except Exception as exc:
            print(f"Error in raster_svg: {exc}")
            import traceback
            traceback.print_exc()
            return [[]], []

    def token_to_color(self, color_token: int) -> str:
        """Convert a raw color-offset token back to a CSS color string."""
        base = self.COLOR_TOKEN_OFFSET
        try:
            if color_token == base:
                return "none"
            if color_token == base + 1:
                return "currentColor"

            color_index = color_token - (base + 2)
            if color_index < 0 or color_index >= self.MAX_COLOR_TOKENS:
                print(f"Warning: Color token {color_token} out of range")
                return "#808080"

            r = (color_index >> 8) & 0xF
            g = (color_index >> 4) & 0xF
            b = color_index & 0xF
            r = (r << 4) | r
            g = (g << 4) | g
            b = (b << 4) | b
            return f"#{r:02x}{g:02x}{b:02x}"
        except Exception as exc:
            print(f"Error in token_to_color: {exc}")
            return "#808080"

    def apply_colors_to_svg(
        self,
        svg_tensors: List[torch.Tensor],
        colors: Optional[List[int]],
    ) -> SVG:
        """Apply colors to SVG path tensors and return a single SVG object."""
        paths: List[SVG] = []

        if not svg_tensors:
            raise ValueError("No valid SVG tensors")

        colors = colors or []

        for i, path_tensor in enumerate(svg_tensors):
            try:
                path = SVGTensor.from_data(path_tensor)
                path = SVG.from_tensor(path.data, viewbox=Bbox(self.BBOX))

                actual_color = (
                    self.token_to_color(colors[i]) if i < len(colors) else "none"
                )
                for path_group in path:
                    path_group.color = actual_color
                    path_group.stroke_color = "none"

                path.fill_(True)
                paths.append(path)
            except Exception as exc:
                print(f"Error processing path {i}: {exc}")
                continue

        if not paths:
            raise ValueError("No valid paths generated")

        path_groups = paths[0].svg_path_groups
        for p in paths[1:]:
            path_groups.extend(p.svg_path_groups)

        return SVG(path_groups, viewbox=Bbox(self.BBOX))