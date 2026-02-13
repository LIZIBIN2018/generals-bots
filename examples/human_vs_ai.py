"""Play against the AI! Human vs ExpanderAgent - Generals.io style controls."""
import pygame
import numpy as np
import jax.numpy as jnp
import jax.random as jrandom
import time

from generals import GeneralsEnv, get_observation
from generals.agents import HumanAgent, ExpanderAgent
from generals.core.game import get_info
from generals.core.rendering import JaxGameAdapter
from generals.gui.gui import GUI
from generals.gui.properties import GuiMode
from generals.core.config import Dimension

# Configuration
GRID_DIMS = (10, 10)
TRUNCATION = 10000  # Max steps (basically unlimited)
GAME_TICK_MS = 250  # Game step every 250ms


def show_start_screen(screen, screen_width, screen_height, result_text=None):
    """Show start screen with controls info. Returns True to play, False to quit."""
    BG_COLOR = (40, 44, 52)
    BUTTON_COLOR = (66, 133, 244)
    BUTTON_HOVER = (100, 160, 255)
    TEXT_COLOR = (255, 255, 255)
    TITLE_COLOR = (255, 200, 100)
    KEY_COLOR = (100, 200, 255)
    WIN_COLOR = (100, 255, 100)
    LOSE_COLOR = (255, 100, 100)
    
    title_font = pygame.font.Font(None, 64)
    button_font = pygame.font.Font(None, 48)
    info_font = pygame.font.Font(None, 26)
    result_font = pygame.font.Font(None, 40)
    
    button_width, button_height = 200, 60
    button_x = (screen_width - button_width) // 2
    button_y = screen_height // 2 + 80
    button_rect = pygame.Rect(button_x, button_y, button_width, button_height)
    
    instructions = [
        ("Controls:", TITLE_COLOR),
        ("", TEXT_COLOR),
        ("  Arrow/WASD  - Move cursor & queue command", TEXT_COLOR),
        ("  Space       - Toggle 50% army mode", TEXT_COLOR),
        ("  Q           - Clear all commands", TEXT_COLOR),
        ("  E           - Undo last command", TEXT_COLOR),
        ("  Click       - Jump cursor", TEXT_COLOR),
        ("  Esc         - Quit game", TEXT_COLOR),
        ("", TEXT_COLOR),
        ("Goal: Capture the enemy general!", KEY_COLOR),
    ]
    
    clock = pygame.time.Clock()
    
    while True:
        mouse_pos = pygame.mouse.get_pos()
        hover = button_rect.collidepoint(mouse_pos)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.MOUSEBUTTONDOWN and hover:
                return True
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_RETURN, pygame.K_SPACE):
                    return True
                if event.key == pygame.K_ESCAPE:
                    return False
        
        screen.fill(BG_COLOR)
        
        # Show result from previous game
        if result_text:
            color = WIN_COLOR if "WON" in result_text else LOSE_COLOR
            result = result_font.render(result_text, True, color)
            screen.blit(result, result.get_rect(center=(screen_width // 2, 25)))
        
        title = title_font.render("Generals.io", True, TITLE_COLOR)
        screen.blit(title, title.get_rect(center=(screen_width // 2, 60 if result_text else 50)))
        
        subtitle = info_font.render("Human vs AI", True, TEXT_COLOR)
        screen.blit(subtitle, subtitle.get_rect(center=(screen_width // 2, 100 if result_text else 90)))
        
        y = 130 if result_text else 120
        for text, color in instructions:
            screen.blit(info_font.render(text, True, color), (30, y))
            y += 26
        
        pygame.draw.rect(screen, BUTTON_HOVER if hover else BUTTON_COLOR, button_rect, border_radius=10)
        pygame.draw.rect(screen, TEXT_COLOR, button_rect, 2, border_radius=10)
        
        button_text = button_font.render("START", True, TEXT_COLOR)
        screen.blit(button_text, button_text.get_rect(center=button_rect.center))
        
        pygame.display.flip()
        clock.tick(30)


def draw_cursor_and_queue(screen, human, cell_size):
    """Draw cursor and queued path."""
    # Draw queued path as small dots
    queue_path = human.get_queue_path()
    for i, (row, col) in enumerate(queue_path):
        x = col * cell_size + cell_size // 2
        y = row * cell_size + cell_size // 2
        color = (255, 255, 100)
        pygame.draw.circle(screen, color, (x, y), 6)
        pygame.draw.circle(screen, (0, 0, 0), (x, y), 6, 1)
    
    # Draw cursor
    row, col = human.get_cursor()
    x = col * cell_size
    y = row * cell_size
    
    if human.split_mode:
        color = (255, 200, 0)
        thickness = 4
    else:
        color = (255, 255, 255)
        thickness = 3
    
    rect = pygame.Rect(x + 2, y + 2, cell_size - 4, cell_size - 4)
    pygame.draw.rect(screen, color, rect, thickness)
    
    # Queue count indicator
    queue_len = len(human.action_queue)
    if queue_len > 0:
        font = pygame.font.Font(None, 24)
        text = font.render(f"Q:{queue_len}", True, (255, 255, 0))
        screen.blit(text, (x + 4, y + 4))
    
    if human.split_mode:
        font = pygame.font.Font(None, 20)
        text = font.render("50%", True, (255, 100, 100))
        screen.blit(text, (x + cell_size - 30, y + 4))


def play_game(screen, cell_size):
    """Play one game. Returns result text or None if quit."""
    # Initialize environment
    env = GeneralsEnv(grid_dims=GRID_DIMS, truncation=TRUNCATION)
    human = HumanAgent(id="You")
    ai = ExpanderAgent(id="AI")
    agent_ids = [human.id, ai.id]
    
    human.set_grid_dims(GRID_DIMS)
    
    key = jrandom.PRNGKey(int(time.time() * 1000) % (2**31))
    state = env.reset(key)
    
    # Setup GUI
    adapter = JaxGameAdapter(state, agent_ids, get_info(state))
    agent_data = {human.id: {"color": "blue"}, ai.id: {"color": "red"}}
    
    pygame.display.set_caption("Generals - Human vs AI")
    gui = GUI(adapter, agent_data, mode=GuiMode.TRAIN)
    renderer = gui._GUI__renderer
    
    human.set_gui_params(cell_size=cell_size, grid_offset=(0, 0))
    
    # Find human's general position and set cursor there
    obs_init = get_observation(state, 0)
    general_pos = np.argwhere(np.array(obs_init.generals))
    if len(general_pos) > 0:
        human.cursor = (int(general_pos[0][0]), int(general_pos[0][1]))
    
    cached_obs = get_observation(state, 0)
    
    terminated = truncated = quit_game = False
    step_count = 0
    last_step_time = pygame.time.get_ticks()
    
    print("\n--- Game Started ---")
    print("Arrow keys = queue command | Q = clear | E = undo | Space = 50%")
    
    while not (terminated or truncated or quit_game):
        current_time = pygame.time.get_ticks()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None  # Quit completely
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return "QUIT"  # Back to menu
                human.handle_event(event, cached_obs, GRID_DIMS)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                human.handle_event(event, cached_obs, GRID_DIMS)
        
        # Game step at fixed intervals
        if current_time - last_step_time >= GAME_TICK_MS:
            last_step_time = current_time
            
            obs_0 = get_observation(state, 0)
            obs_1 = get_observation(state, 1)
            
            key, k1, k2 = jrandom.split(key, 3)
            action_human = human.act(obs_0, k1)
            action_ai = ai.act(obs_1, k2)
            actions = jnp.stack([action_human, action_ai])
            
            timestep, state = env.step(state, actions)
            
            cached_obs = get_observation(state, 0)
            adapter.update_from_state(state, timestep.info)
            
            terminated = bool(timestep.terminated)
            truncated = bool(timestep.truncated)
            step_count += 1
        
        # Render
        renderer.render()
        draw_cursor_and_queue(renderer.screen, human, cell_size)
        pygame.display.flip()
        
        pygame.time.wait(16)
    
    # Return result
    if timestep.info.winner == 0:
        result = f"You WON! ({step_count} steps)"
        print(f"\n*** {result} ***")
    elif timestep.info.winner == 1:
        result = f"You lost... ({step_count} steps)"
        print(f"\n{result}")
    else:
        result = f"Draw ({step_count} steps)"
        print(f"\n{result}")
    
    time.sleep(1.5)
    return result


def main():
    print("=" * 50)
    print("   Generals.io - Human vs AI")
    print("=" * 50)
    
    pygame.init()
    
    cell_size = Dimension.SQUARE_SIZE.value
    screen_width = GRID_DIMS[1] * cell_size + 200
    screen_height = GRID_DIMS[0] * cell_size + 50
    
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Generals - Human vs AI")
    
    # Warm up JAX once
    print("Initializing JAX...")
    env = GeneralsEnv(grid_dims=GRID_DIMS, truncation=100)
    key = jrandom.PRNGKey(0)
    state = env.reset(key)
    _ = get_observation(state, 0)
    dummy_actions = jnp.zeros((2, 5), dtype=jnp.int32)
    _ = env.step(state, dummy_actions)
    print("Ready!")
    
    result_text = None
    
    # Main loop - show start screen, play game, repeat
    while True:
        # Show start screen
        if not show_start_screen(screen, screen_width, screen_height, result_text):
            break  # User wants to quit
        
        # Play game
        result = play_game(screen, cell_size)
        
        if result is None:
            break  # Window closed
        elif result == "QUIT":
            result_text = None  # Escaped, no result to show
        else:
            result_text = result  # Show result on start screen
    
    pygame.quit()
    print("\nThanks for playing!")


if __name__ == "__main__":
    main()
