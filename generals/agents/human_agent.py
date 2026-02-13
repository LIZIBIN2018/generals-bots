"""Human-controlled agent using keyboard input - Generals.io style controls."""
from collections import deque
import pygame
import numpy as np
import jax.numpy as jnp

from generals.core.observation import Observation
from .agent import Agent


# Pre-create pass action
PASS_ACTION = np.array([1, 0, 0, 0, 0], dtype=np.int32)


class HumanAgent(Agent):
    """
    Human-controlled agent with Generals.io style controls.
    
    Controls:
        - Arrow keys / WASD: Move cursor AND queue move command
        - Space (double tap): Toggle split mode (50% army)
        - Q: Clear all queued commands
        - E: Undo last queued command
        - Click: Jump cursor to cell
    
    The cursor can move freely to plan paths. Commands are queued
    based on cursor position, not actual army position.
    """

    def __init__(self, id: str = "Human"):
        super().__init__(id)
        self.cursor = (0, 0)  # Current cursor position (row, col)
        self.action_queue = deque()  # Queue of (action, cursor_pos) tuples
        self.cell_size = 50
        self.grid_dims = (10, 10)
        self.split_mode = False  # 50% army mode
        self.last_space_time = 0  # For double-tap detection
        
    def set_gui_params(self, cell_size: int, grid_offset: tuple):
        """Set GUI parameters."""
        self.cell_size = cell_size

    def set_grid_dims(self, grid_dims: tuple):
        """Set grid dimensions and initialize cursor to center."""
        self.grid_dims = grid_dims
        self.cursor = (grid_dims[0] // 2, grid_dims[1] // 2)

    def act(self, observation: Observation, key: jnp.ndarray) -> np.ndarray:
        """Get action from queue, or pass if empty."""
        if self.action_queue:
            action, _ = self.action_queue.popleft()
            return action
        return PASS_ACTION.copy()

    def get_cursor(self) -> tuple:
        """Get current cursor position for rendering."""
        return self.cursor

    def get_queue_path(self) -> list:
        """Get list of positions in the queue for rendering."""
        return [pos for _, pos in self.action_queue]

    def handle_event(self, event: pygame.event.Event, observation: Observation, grid_dims: tuple):
        """Handle pygame events."""
        self.grid_dims = grid_dims
        
        if event.type == pygame.KEYDOWN:
            self._handle_key_press(event, observation)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            self._handle_mouse_click(event, observation)

    def _handle_mouse_click(self, event: pygame.event.Event, observation: Observation):
        """Handle mouse click - jump cursor to clicked cell."""
        x, y = event.pos
        col = x // self.cell_size
        row = y // self.cell_size
        
        h, w = self.grid_dims
        if 0 <= row < h and 0 <= col < w:
            self.cursor = (int(row), int(col))
            print(f"Cursor: ({row},{col})")

    def _handle_key_press(self, event: pygame.event.Event, observation: Observation):
        """Handle key press - Generals.io style."""
        # Direction mapping: 0=up, 1=down, 2=left, 3=right
        key_to_direction = {
            pygame.K_UP: 0, pygame.K_w: 0,
            pygame.K_DOWN: 1, pygame.K_s: 1,
            pygame.K_LEFT: 2, pygame.K_a: 2,
            pygame.K_RIGHT: 3, pygame.K_d: 3,
        }
        
        direction = key_to_direction.get(event.key)
        
        # Q key - clear queue
        if event.key == pygame.K_q:
            count = len(self.action_queue)
            self.action_queue.clear()
            self.split_mode = False
            print(f"Queue cleared ({count} commands)")
            return
        
        # E key - undo last command
        if event.key == pygame.K_e:
            if self.action_queue:
                _, last_pos = self.action_queue.pop()
                # Move cursor back to where that command was from
                self.cursor = last_pos
                print(f"Undo -> ({last_pos[0]},{last_pos[1]}) [q={len(self.action_queue)}]")
            else:
                print("Nothing to undo")
            return
        
        # Escape - clear queue and reset
        if event.key == pygame.K_ESCAPE:
            self.action_queue.clear()
            self.split_mode = False
            print("Reset")
            return
        
        # Space - toggle split mode (double tap style)
        if event.key == pygame.K_SPACE:
            # Toggle split mode
            self.split_mode = not self.split_mode
            status = "ON (50%)" if self.split_mode else "OFF"
            print(f"Split: {status}")
            return
        
        # Direction key - queue move and advance cursor
        if direction is not None:
            self._queue_move(direction)

    def _queue_move(self, direction: int):
        """Queue a move command from current cursor position and advance cursor."""
        row, col = self.cursor
        
        # Calculate target position
        dr = [-1, 1, 0, 0][direction]
        dc = [0, 0, -1, 1][direction]
        new_row = row + dr
        new_col = col + dc
        
        # Check bounds
        h, w = self.grid_dims
        if not (0 <= new_row < h and 0 <= new_col < w):
            print(f"Can't move: out of bounds")
            return
        
        # Create action: [pass=0, row, col, direction, split]
        split = 1 if self.split_mode else 0
        action = np.array([0, row, col, direction, split], dtype=np.int32)
        
        # Store action with source position (for undo)
        self.action_queue.append((action, (row, col)))
        
        # Move cursor to target
        self.cursor = (new_row, new_col)
        
        dir_names = ['↑', '↓', '←', '→']
        suffix = " 50%" if split else ""
        print(f"({row},{col}){dir_names[direction]}{suffix} [q={len(self.action_queue)}]")

    def reset(self):
        """Reset agent state."""
        self.cursor = (self.grid_dims[0] // 2, self.grid_dims[1] // 2)
        self.split_mode = False
        self.action_queue.clear()
