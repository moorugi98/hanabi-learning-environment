import copy
import numpy as np
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
        prior[plyr, -1] = [0.3, 0.2, 0.5]  # agnostic for new card

    # Get intention for each card independently
    table = np.zeros((game.num_players(), game.hand_size(), 3))
    # TODO: exclude active player for the actual use?
    realisations = get_realisations_probs(game, knowledge, sample_num=100)  # joint realisations is same for all cards
    for pi in range(game.num_players()):
        for i in range(game.hand_size()):
            table[pi, i] = pragmatic_listener(game, action, state, prior[pi, i], realisations)
    return table  # dim: (num_plyr, num_hand, 3)


def get_realisations_probs(game, knowledge, sample_num=100):
    """
    Returns a list of tuples.
    The first element is the realisation of all cards dim: [plyr, card, 2] with
    0-th elem being color, 1-st elem being rank.
    The second element is the probability to get that realisation, P(r|c)
    To make the problem tractable, sample @sample_num times and count frequency
    """

    # Get realisation for each individual card
    total_r = []  # nested list with tuples of realisations for each plyr for each card
    total_p = []
    for player_index in range(game.num_players()):
        single_plyr_r = []
        single_plyr_p = []
        for card_index in range(game.hand_size()):
            single_card_r = []
            single_card_p = []
            for col in range(game.num_colors()):
                for rank in range(game.num_ranks()):
                    # realisations that are not possible
                    if knowledge[player_index][card_index][col][rank] == 0:
                        pass
                    else:
                        single_card_r.append([col,rank])
                        single_card_p.append(knowledge[player_index][card_index][col][rank]
                                             / np.sum(knowledge[player_index][card_index]))
            single_plyr_r.append(single_card_r)
            single_plyr_p.append(single_card_p)
        total_r.append(single_plyr_r)
        total_p.append(single_plyr_p)

    # Combine individual realisation to get a joint realisation by sampling sample_num times
    samples = []  # nested list with dim: [num_sample, num_plyr, num_card]
    for _ in range(sample_num):
        joint_sample = []
        for player_index in range(game.num_players()):
            plyr_sample = []
            for card_index in range(game.hand_size()):
                card_sample = np.random.choice(total_r[player_index][card_index], p=total_p[player_index][card_index])
                plyr_sample.append(card_sample)
            joint_sample.append(plyr_sample)
        samples.append(joint_sample)

    # Normalize frequency to get probability
    realisations, counts = np.unique(samples, return_counts=True, axis=0)
    return realisations, counts / np.sum(counts)


def pragmatic_listener(game, action, state, prior, realisations):
    '''
    return a 3 dim simplex for PLAY,DISCARD,KEEP
    '''

    # 3 dim simplex with prob for each intention
    probs = []

    # 3 different numerators for each intention
    for intention in range(3):
        numerator = 0
        # sum over r which is given as argument
        for r, p in realisations:
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
    alpha = 10

    # compute numerator
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
    # TODO: legal_moves() should come from realization-specific states, see line 494 in pyhanabi.py

    # print("!!!!!!!!!!!!!!!")
    # a = move.get_reveal_rank_move(target_offset=2, rank=0)
    # print(a)
    # state.apply_move(a)
    # print("!!!!!!!!!!!!")
    # print(state)
    # break

    for a in state.legal_moves():
        ns = state.copy()
        # this is how different actions makes a difference
        ns.apply_move(a)
        nk = generate_knowledge(game, ns)
        denominator += np.exp(alpha * utility(intention, realisation, ns, nk))

    return numerator / denominator
##########################################################################
