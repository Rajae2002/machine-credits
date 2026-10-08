"""Lance l'entraînement du Double Q-Learning : python -m training.train_double_q_learning --episodes 20000 --seed 0"""
from training.trainer import main

if __name__ == "__main__":
    main("double_q_learning")
