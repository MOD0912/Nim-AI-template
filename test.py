from nim import NimAI

def test_get_q_value(ai):
    print("\n--- Testing get_q_value ---")
    state = (0, 0, 0, 2)
    action = (3, 2)
    value = ai.get_q_value(state, action)
    print(f"Q-value for state {state}, action {action}: {value}")


def test_update_q_value(ai):
    print("\n--- Testing update_q_value ---")
    ai.update_q_value((0, 0, 0, 2), (3, 2), -1, 10, 5)


def test_best_future_reward(ai):
    print("\n--- Testing best_future_reward ---")
    future = ai.best_future_reward((0, 0, 0, 2))
    print(future)


def test_choose_action(ai):
    print("\n--- Testing choose_action ---")
    action = ai.choose_action((0, 0, 0, 2))
    print(f"Chosen action: {action}")


if __name__ == "__main__":
    ai = NimAI()

    test_get_q_value(ai)
    test_update_q_value(ai)
    test_best_future_reward(ai)
    test_choose_action(ai)

    print("\nAll tests completed.")
