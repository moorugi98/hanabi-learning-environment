# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example code demonstrating the Python Hanabi interface."""

from __future__ import print_function
import os
from datetime import datetime
import numpy as np
from hanabi_learning_environment import pyhanabi
import intention_update


def run_game(game_parameters):
    """Play a game, selecting random actions."""

    game = pyhanabi.HanabiGame(game_parameters)
    print(game.parameter_string(), end="")

    state = game.new_initial_state()
    counter = 0
    # initial intention is agnostic for PLAY, DISCARD, KEEP
    intention = np.array([[[0.33, 0.33, 0.34] for i in range(game.hand_size())] for pi in range(game.num_players())])
    intention_history = []

    while not state.is_terminal():
        if state.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
            state.deal_random_card()
            continue

        print()
        print()
        print()
        print("counter: ", counter)
        print("STATE")
        print(state)

        print("KNOWLEDGE")
        knowledge = intention_update.generate_knowledge(game, state)
        print(knowledge)

        legal_moves = state.legal_moves()
        print("")
        print("Number of legal moves: {}".format(len(legal_moves)))
        move = np.random.choice(legal_moves)
        print("Chose random legal move: {}".format(move))
        # make screenshot of old state before apply the move
        old_state = state.copy()
        state.apply_move(move)

        # code intentions
        print()
        print("INTENTION")
        intention = intention_update.infer_joint_intention(
                                                        game=game,
                                                        action=move,
                                                        state=old_state,
                                                        knowledge=knowledge,
                                                        prior=intention)
        intention_history.append(intention)
        np.set_printoptions(precision=2, suppress=True)
        print(intention)
        counter += 1

    print("")
    print("Game done. Terminal state:")
    print("")
    print(state)
    print("")
    print("score: {}".format(state.score()))
    path = os.path.join(os.getcwd(), 'history', 'score:{}_{}.npy'.format(state.score(), datetime.now()))
    np.save(path, intention_history)  # save histories of intention change for later analysis


if __name__ == "__main__":
    # Check that the cdef and library were loaded from the standard paths.
    assert pyhanabi.cdef_loaded(), "cdef failed to load"
    assert pyhanabi.lib_loaded(), "lib failed to load"
    # currently, only using colors=4 ensures correct return of firework after copying the state due to bug
    run_game({"players": 3, "hand_size": 2, "colors": 4, "random_start_player": False})
