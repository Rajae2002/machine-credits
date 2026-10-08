"""Lance l'entraînement du Q-Learning : python -m training.train_q_learning --episodes 20000 --seed 0"""
from training.trainer import main

if __name__ == "__main__":
    main("q_learning")
