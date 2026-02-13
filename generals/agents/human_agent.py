"""Human-controlled agent using keyboard input."""
from collections import deque
import pygame
import numpy as np
import jax.numpy as jnp

from generals.core.observation import Observation
from .agent import Agent


# Pre-create pass action (avoid repeated JAX array creation)
PASS_ACTION = np.array([1, 0, 0, 0, 0], dtype=np.int32)


class HumanAgent(Agent):
    """
    Human-controlled agent using keyboard and mouse.
    
    Controls:
        - Arrow keys / WASD: Move direction
        - Click: Select source cell
        - Space: Pass turn
        - Shift + direction: Split army (send half)
    
    Commands are queued and executed one per game step.
    """

    def __init__(self, id: str = "Human"):
        super().__init__(id)
        self.selected_cell = None  # (row, col) or None
        self.action_queue = deque()  # Queue of actions (numpy arrays)
        self.cell_size = 50  # Will be updated by GUI
        self.grid_offset = (0, 0)  # Will be updated by GUI
        
    def set_gui_params(self, cell_size: int, grid_offset: tuple):
        """Set GUI parameters for mouse click translation."""
        self.cell_size = cell_size
        self.grid_offset = grid_offset

    def act(self, observation: Observation, key: jnp.ndarray) -> jnp.ndarray:
        """
        Get action from queue, or pass if queue is empty.
        """
        if self.action_queue:
            action = self.action_queue.popleft()
            return jnp.asarray(action)  # Convert to JAX array only when needed
        
        # Default: pass
        return jnp.asarray(PASS_ACTION)

    def handle_event(self, event: pygame.event.Event, observation: Observation, grid_dims: tuple):
        """
        Handle pygame events for human input.
        
        Args:
            event: pygame event
            observation: current observation
            grid_dims: (height, width) of the grid
        """
        if event.type == pygame.MOUSEBUTTONDOWN:
            self._handle_mouse_click(event, observation, grid_dims)
        elif event.type == pygame.KEYDOWN:
            self._handle_key_press(event, observation)

    def _handle_mouse_click(self, event: pygame.event.Event, observation: Observation, grid_dims: tuple):
        """Handle mouse click to select a cell."""
        x, y = event.pos
        
        # Convert pixel position to grid cell
        col = (x - self.grid_offset[0]) // self.cell_size
        row = (y - self.grid_offset[1]) // self.cell_size
        
        # Check if click is within grid bounds
        if 0 <= row < grid_dims[0] and 0 <= col < grid_dims[1]:
            # Check if we own this cell and have armies > 1
            if observation.owned_cells[row, col] and observation.armies[row, col] > 1:
                self.selected_cell = (int(row), int(col))
                print(f"Selected: ({row}, {col}) army={int(observation.armies[row, col])}")
            else:
                self.selected_cell = None

    def _handle_key_press(self, event: pygame.event.Event, observation: Observation):
        """Handle key press for movement."""
        # Ignore modifier keys
        if event.key in (pygame.K_LSHIFT, pygame.K_RSHIFT, pygame.K_LCTRL, pygame.K_RCTRL,
                         pygame.K_LALT, pygame.K_RALT, pygame.K_LMETA, pygame.K_RMETA):
            return
        
        if self.selected_cell is None:
            if event.key == pygame.K_SPACE:
                self.action_queue.append(PASS_ACTION.copy())
                print("Queued: PASS")
            return

        row, col = self.selected_cell
        direction = None
        
        # Direction mapping: 0=up, 1=down, 2=left, 3=right
        key_to_direction = {
            pygame.K_UP: 0, pygame.K_w: 0,
            pygame.K_DOWN: 1, pygame.K_s: 1,
            pygame.K_LEFT: 2, pygame.K_a: 2,
            pygame.K_RIGHT: 3, pygame.K_d: 3,
        }
        
        if event.key in key_to_direction:
            direction = key_to_direction[event.key]
        elif event.key == pygame.K_SPACE:
            self.action_queue.append(PASS_ACTION.copy())
            self.selected_cell = None
            print("Queued: PASS")
            return
        elif event.key == pygame.K_ESCAPE:
            self.selected_cell = None
            return
        else:
            return  # Unknown key, ignore
        
        if direction is not None:
            # Check if shift is held for split
            mods = pygame.key.get_mods()
            split = 1 if mods & pygame.KMOD_SHIFT else 0
            
            # Create action using numpy (fast): [pass=0, row, col, direction, split]
            action = np.array([0, row, col, direction, split], dtype=np.int32)
            self.action_queue.append(action)
            
            dir_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
            print(f"Queued: ({row},{col})->{dir_names[direction]} [q={len(self.action_queue)}]")
            
            # Move selection to target cell for continuous movement
            dr = [-1, 1, 0, 0][direction]
            dc = [0, 0, -1, 1][direction]
            new_row, new_col = row + dr, col + dc
            
            # Update selection to new cell for chaining moves
            h, w = observation.owned_cells.shape
            if 0 <= new_row < h and 0 <= new_col < w:
                self.selected_cell = (new_row, new_col)
            else:
                self.selected_cell = None

    def reset(self):
        """Reset agent state."""
        self.selected_cell = None
        self.action_queue.clear()
