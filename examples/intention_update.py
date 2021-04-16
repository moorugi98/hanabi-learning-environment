import copy
import numpy as np
# TODO: select the right utility function
# from utility_Saskia import utility
from Edited_Utility_Function_Bianca import utility

# # encode constants here temporally for convenience although ideally it should be encoded in pyhanabi.py
COUNTS = [3, 2, 2, 2, 1]

################# Init & update knowledge ####################
def generate_knowledge(game, state):
    # Initial knowledge is full [3,2,2,2,1] for each rank
    knowledge = np.array(
        [
            [
                [COUNTS[: game.num_ranks()] for col in range(game.num_colors())]
                for card in range(game.hand_size())
            ]
            for plyr in range(game.num_players())
        ]
    )

    # Subtract discarded cards
    for card in state.discard_pile():
        knowledge[:, :, card.color(), card.rank()] -= 1

    # Subtract already played cards
    for color, stack in enumerate(state.fireworks()):  # R:3, G:2,...
        for rank in range(stack):
            knowledge[:, :, color, rank] -= 1

    # Use public knowledge to set any impossible realizations to 0
    obs = state.observation(0)  # accessed via player's observation although it is public information
    hck = obs.card_knowledge()  # I don't know why, but doing this is one-liner causes weird problems
    for pi, player in enumerate(hck):
        for hi, hand in enumerate(player):
            # reduce color possibility
            for color in range(game.num_colors()):
                if not hand.color_plausible(color):
                    knowledge[pi, hi, color] = 0
            # reduce rank possibility
            for rank in range(game.num_ranks()):
                if not hand.rank_plausible(rank):
                    knowledge[pi, hi, :, rank] = 0

            # When the card is known for sure, reduce count by one for all other cards
            if np.sum(knowledge[pi, hi] > 0) == 1:
                col, rank = np.nonzero(knowledge[pi, hi] > 0)
                # reduce the possibility for all other cards except the realisation itself
                knowledge[:, :, col[0], rank[0]] = np.maximum(
                    knowledge[:, :, col[0], rank[0]] - 1, 0
                )
                knowledge[pi, hi, col[0], rank[0]] += 1
    return knowledge
#######################################################################################


################################################## INTENTION UPDATE #############################################
def infer_joint_intention(game, action, state, knowledge, prior):
    # If action was discard or play, shift card indices and set intention for the new card
    obs = state.observation(0)
    if (obs.last_moves() != []) and (  # skip the very first move since no last move
            obs.last_moves()[0].move().type() == 5):  # if the last move was dealing the card
        plyr = obs.last_moves()[1].player()
        for i in range(obs.last_moves()[1].move().card_index(), game.hand_size()-1):  # shift by one
            prior[plyr, i] = copy.deepcopy(prior[plyr, i+1])
        prior[plyr, -1] = [0.33, 0.33, 0.34]  # agnostic for new card

    # Get intention for each card independently
    table = np.zeros((game.num_players(), game.hand_size(), 3))
    for pi in range(game.num_players()):
        for i in range(game.hand_size()):
            table[pi,i] = pragmatic_listener(game, action, state, knowledge, pi, i, prior[pi,i])
    return table  # dim: (num_plyr, num_hand, 3)


def get_realisations_probs(game, knowledge, player_index, card_index):
    """
    returns a list of tuples with the first element being the realisation of a single card (type dictionary)
    and the second element being the probability to get that realisation, P(r|c)
    """

    mylist = []
    for col in range(game.num_colors()):
        for rank in range(game.num_ranks()):
            # realisations that are not possible
            if knowledge[player_index][card_index][col][rank] == 0:
                pass
            else:
                mylist.append(
                    (
                        {"color": col, "rank": rank},
                        knowledge[player_index][card_index][col][rank]
                        / np.sum(knowledge[player_index][card_index]),
                    )
                )
    return mylist


def pragmatic_listener(game, action, state, knowledge, player_index, card_index, prior):
    '''
    return a 3 dim simplex for PLAY,DISCARD,KEEP
    '''

    # 3 dim simplex with prob for each intention
    probs = []

    # 3 different numerators for each intention
    for intention in range(3):
        numerator = 0
        # sum over r
        for r, p in get_realisations_probs(game, knowledge, player_index, card_index):
            numerator += pragmatic_speaker(game, action, intention, r, state) * \
                     prior[intention] * p
        # save each value
        probs.append(numerator)

    # normalise to probability distribution
    return probs / np.sum(probs)



def pragmatic_speaker(game, action, intention, realisation, state):
    """
    return a scala which is P(action|intention,realisation,context)
    """
    # TODO: adjust rationality parameter dynamically
    alpha = 1

    # compute numerator
    # TODO: copying doesn't perfectly copy??  state.fireworks()
    new_state = state.copy()
    new_state.apply_move(action)
    new_knowledge = generate_knowledge(game, new_state)
    numerator = np.exp(
        alpha * utility(intention, realisation, new_state, new_knowledge)
    )

    # compute denominator
    denominator = 0
    # automatically only select actions with P(a*|r,c) != 0
    # iterate over all actions that the last agent could've taken
    for a in state.legal_moves():
        ns = state.copy()
        # this is how different actions makes a difference
        ns.apply_move(a)
        nk = generate_knowledge(game, ns)
        denominator += np.exp(alpha * utility(intention, realisation, ns, nk))

    return numerator / denominator
##########################################################################
