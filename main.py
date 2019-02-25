"""Main file of project.

Run this file for schedulability analysis with recurrent neural network (RNN).
"""
from sklearn.model_selection import train_test_split

from database import Database
from models import Single_LSTM_Model, Single_GRU_Model


def main():
    """Main function."""
    my_database = Database()
    features, labels = my_database.read_all_tasksets()
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2,
                                                        shuffle=False, random_state=42)

    print("-" * 10, " Single LSTM Model ", "-" * 10)
    lstm_model = Single_LSTM_Model()
    lstm_model.train_neural_network(X_train, y_train, X_test, y_test)
    print("-" * 40)

    print("-" * 10, " Single GRU Model ", "-" * 10)
    gru_model = Single_GRU_Model()
    gru_model.train_neural_network(X_train, y_train, X_test, y_test)
    print("-" * 40)


if __name__ == "__main__":
    main()
