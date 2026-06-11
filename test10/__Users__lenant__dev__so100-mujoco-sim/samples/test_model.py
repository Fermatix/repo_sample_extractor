import os
import gymnasium as gym
import time

# Import our custom environment (required for env registration)
import custom_reach_env

def test_model(model_path="models/reach_final_model.zip", episodes=10, steps_per_episode=100):
    """
    Load and test a trained model in the Reach environment
    
    Args:
        model_path: Path to the saved model file
        episodes: Number of episodes to run
        steps_per_episode: Maximum steps per episode
    """
    # Check if model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    print(f"Loading model from {model_path}")
    
    # Determine which algorithm to use based on the model filename
    if "sac" in model_path.lower():
        from stable_baselines3 import SAC
        model = SAC.load(model_path)
    else:
        from stable_baselines3 import PPO
        model = PPO.load(model_path)
    
    # Create the environment with rendering
    env = gym.make('Reach-v0', render_mode="human")
    
    # Run test episodes
    for episode in range(episodes):
        print(f"Episode {episode+1}/{episodes}")
        
        # Reset environment
        obs, _ = env.reset()
        episode_reward = 0
        
        # Run episode
        for step in range(steps_per_episode):
            # Get action from model
            action, _states = model.predict(obs, deterministic=True)
            
            # Apply action to environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            
            # Small delay to make visualization easier to follow
            time.sleep(0.01)
            
            # Check if episode is done
            if terminated or truncated:
                print(f"  Episode ended after {step+1} steps with reward {episode_reward:.2f}")
                break
        
        # If episode didn't terminate naturally
        else:
            print(f"  Episode reached max steps with reward {episode_reward:.2f}")
    
    # Clean up
    env.close()
    print("Testing complete!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test a trained Reach environment model")
    parser.add_argument("--model", type=str, default="models/reach_sac_model_1900000_steps.zip", 
                        help="Path to the trained model file")
    parser.add_argument("--episodes", type=int, default=50, 
                        help="Number of episodes to run")
    parser.add_argument("--steps", type=int, default=200, 
                        help="Maximum steps per episode")
    
    args = parser.parse_args()
    
    test_model(args.model, args.episodes, args.steps) 