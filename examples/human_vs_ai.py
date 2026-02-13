"""Play against the AI! Human vs ExpanderAgent - Optimized version."""
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
TRUNCATION = 500
GAME_TICK_MS = 250  # Game step every 250ms (4 steps per second)


def show_start_screen(screen_width, screen_height):
    """Show start screen and wait for player to click Start."""
    pygame.init()
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Generals - Human vs AI")
    
    BG_COLOR = (40, 44, 52)
    BUTTON_COLOR = (66, 133, 244)
    BUTTON_HOVER = (100, 160, 255)
    TEXT_COLOR = (255, 255, 255)
    TITLE_COLOR = (255, 200, 100)
    
    title_font = pygame.font.Font(None, 64)
    button_font = pygame.font.Font(None, 48)
    info_font = pygame.font.Font(None, 28)
    
    button_width, button_height = 200, 60
    button_x = (screen_width - button_width) // 2
    button_y = screen_height // 2
    button_rect = pygame.Rect(button_x, button_y, button_width, button_height)
    
    instructions = [
        "Controls:",
        "  Click - Select your cell",
        "  WASD/Arrows - Move army",
        "  Shift+Move - Split army",
        "  Q - Quit",
        "",
        "Goal: Capture the enemy general!"
    ]
    
    clock = pygame.time.Clock()
    
    while True:
        mouse_pos = pygame.mouse.get_pos()
        hover = button_rect.collidepoint(mouse_pos)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
            if event.type == pygame.MOUSEBUTTONDOWN and hover:
                return True
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_RETURN, pygame.K_SPACE):
                    return True
                if event.key == pygame.K_q:
                    pygame.quit()
                    return False
        
        screen.fill(BG_COLOR)
        
        title = title_font.render("Generals.io", True, TITLE_COLOR)
        screen.blit(title, title.get_rect(center=(screen_width // 2, 80)))
        
        subtitle = info_font.render("Human vs AI", True, TEXT_COLOR)
        screen.blit(subtitle, subtitle.get_rect(center=(screen_width // 2, 120)))
        
        y = 160
        for line in instructions:
            screen.blit(info_font.render(line, True, TEXT_COLOR), (50, y))
            y += 28
        
        pygame.draw.rect(screen, BUTTON_HOVER if hover else BUTTON_COLOR, button_rect, border_radius=10)
        pygame.draw.rect(screen, TEXT_COLOR, button_rect, 2, border_radius=10)
        
        button_text = button_font.render("START", True, TEXT_COLOR)
        screen.blit(button_text, button_text.get_rect(center=button_rect.center))
        
        hint = info_font.render("Press ENTER to begin", True, (150, 150, 150))
        screen.blit(hint, hint.get_rect(center=(screen_width // 2, button_y + 100)))
        
        pygame.display.flip()
        clock.tick(30)


def main():
    print("=" * 50)
    print("   Generals.io - Human vs AI")
    print("=" * 50)
    
    cell_size = Dimension.SQUARE_SIZE.value
    screen_width = GRID_DIMS[1] * cell_size + 200
    screen_height = GRID_DIMS[0] * cell_size + 50
    
    if not show_start_screen(screen_width, screen_height):
        return
    
    # Initialize environment
    env = GeneralsEnv(grid_dims=GRID_DIMS, truncation=TRUNCATION)
    human = HumanAgent(id="You")
    ai = ExpanderAgent(id="AI")
    agent_ids = [human.id, ai.id]
    
    key = jrandom.PRNGKey(int(time.time()))  # Random seed
    state = env.reset(key)
    
    # Warm up JAX (compile once)
    print("Initializing...")
    _ = get_observation(state, 0)
    _ = get_observation(state, 1)
    dummy_actions = jnp.zeros((2, 5), dtype=jnp.int32)
    _ = env.step(state, dummy_actions)
    print("Ready!")
    
    # Setup GUI
    adapter = JaxGameAdapter(state, agent_ids, get_info(state))
    agent_data = {human.id: {"color": "blue"}, ai.id: {"color": "red"}}
    
    pygame.display.set_caption("Generals - Human vs AI")
    gui = GUI(adapter, agent_data, mode=GuiMode.TRAIN)
    renderer = gui._GUI__renderer
    
    human.set_gui_params(cell_size=cell_size, grid_offset=(0, 0))
    
    # Cache observation for input handling (updated after each game step)
    cached_obs = get_observation(state, 0)
    
    terminated = truncated = quit_game = False
    step_count = 0
    last_step_time = pygame.time.get_ticks()
    
    # Initial render
    renderer.render()
    
    while not (terminated or truncated or quit_game):
        current_time = pygame.time.get_ticks()
        
        # Handle input (fast, doesn't call JAX)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_game = True
                break
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    quit_game = True
                    break
                human.handle_event(event, cached_obs, GRID_DIMS)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                human.handle_event(event, cached_obs, GRID_DIMS)
        
        if quit_game:
            break
        
        # Game step (only at fixed intervals)
        if current_time - last_step_time >= GAME_TICK_MS:
            last_step_time = current_time
            
            # Get observations
            obs_0 = get_observation(state, 0)
            obs_1 = get_observation(state, 1)
            
            # Get actions
            key, k1, k2 = jrandom.split(key, 3)
            action_human = human.act(obs_0, k1)
            action_ai = ai.act(obs_1, k2)
            actions = jnp.stack([action_human, action_ai])
            
            # Step
            timestep, state = env.step(state, actions)
            
            # Update cache and adapter
            cached_obs = get_observation(state, 0)
            adapter.update_from_state(state, timestep.info)
            
            terminated = bool(timestep.terminated)
            truncated = bool(timestep.truncated)
            step_count += 1
        
        # Render (always, for smooth visuals)
        renderer.render()
        pygame.time.wait(16)  # ~60 FPS render
    
    # Game over
    print()
    if not quit_game:
        if timestep.info.winner == 0:
            print("Congratulations! You WON!")
        elif timestep.info.winner == 1:
            print("You lost... The AI won this time.")
        else:
            print("Draw (timeout).")
        print(f"Game lasted {step_count} steps.")
        time.sleep(2)
    
    pygame.quit()


if __name__ == "__main__":
    main()
