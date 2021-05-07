import copy
import random
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
    realisations, r_probs = get_realisations_probs(game, knowledge, sample_num=100)  # joint realisations is same for all cards
    for pi in range(game.num_players()):
        for ci in range(game.hand_size()):
            #TODO: debug
            # print()
            # print()
            # print(pi,ci)
            # print()
            table[pi, ci] = pragmatic_listener(game, action, state, prior[pi, ci], realisations, r_probs, pi, ci)
    return table  # dim: (num_plyr, num_hand, 3)


def get_realisations_probs(game, knowledge, sample_num=100):
    """
    Returns a list of tuples.
    The first element is the realisation of all cards dim: [plyr, card, 2] with
    0-th elem being color, 1-st elem being rank.
    The second element is the probability to get that realisation, P(r|c)
    To make the problem tractable, sample @sample_num times and count frequency
    """
    # Default meshgrid of colors and ranks to be sampled
    card_meshgrid = np.zeros((game.num_colors(), game.num_ranks(), 2))
    card_meshgrid[:, :, 0] = np.repeat(np.array([range(game.num_colors())]).T, game.num_ranks(), axis=1)  # color
    card_meshgrid[:, :, 1] = np.repeat(np.array([range(game.num_ranks())]), game.num_colors(), axis=0)  # rank
    card_meshgrid = np.reshape(card_meshgrid, (-1,2))

    # Sampling
    samples = []  # nested list with dim: [num_sample, num_plyr, num_card, 2 (col,rank)]
    for _ in range(sample_num):
        joint_sample = []
        # sample knowledge diverge whenever a card is sampled, which is subtracted
        sample_knowledge = np.zeros((game.num_colors(), game.num_ranks()))
        for player_index in range(game.num_players()):
            plyr_sample = []
            for card_index in range(game.hand_size()):
                # sample from 1-D array (index array) with appropriate weights
                try:
                    index_to_sample = random.choices(range(game.num_colors()*game.num_ranks()),
                                                 weights=knowledge[player_index][card_index].flatten() - sample_knowledge.flatten())
                except:
                    print("look line 111 of intention update.py")
                    print(player_index, card_index)
                    print(sample_knowledge)
                    exit()
                card_sample = np.int_(card_meshgrid[index_to_sample][0])
                # TODO: always going thru same sequence introduce bias (1st plyr more likely to get a rare card...)
                sample_knowledge[card_sample[0], card_sample[1]] += 1  # now this is sampled
                plyr_sample.append(card_sample)
            joint_sample.append(plyr_sample)
        samples.append(joint_sample)

    # Normalize frequency to get probability
    samples = np.array(samples)
    realisations, counts = np.unique(samples, return_counts=True, axis=0)
    return realisations, counts / np.sum(counts)


def pragmatic_listener(game, action, state, prior, realisations, r_probs, plyr_index, card_index):
    '''
    return a 3 dim simplex for PLAY,DISCARD,KEEP
    '''

    # 3 dim simplex with prob for each intention
    probs = []

    # 3 different numerators for each intention
    for intention in range(3):
        numerator = 0
        # sum over r which is given as argument
        for i, (r, p) in enumerate(zip(realisations, r_probs)):
            #TODO: debug enumerate and print
            # print('realization index: ', i)
            # print('realization: ', r)
            numerator += pragmatic_speaker(game, action, intention, r, state, plyr_index, card_index) * \
                     prior[intention] * p
        # save each value
        probs.append(numerator)

    # normalise to probability distribution
    return probs / np.sum(probs)



def pragmatic_speaker(game, action, intention, realisation, state, plyr_index, card_index):
    """
    return a scala which is P(action|intention,realisation,context)
    """
    # TODO: adjust rationality parameter dynamically
    alpha = 10

    # Compute numerator
    new_state = state.copy()
    new_state.apply_move(action)
    new_knowledge = generate_knowledge(game, new_state)
    numerator = np.exp(
        alpha * utility(intention, realisation[plyr_index][card_index], new_state, new_knowledge)
    )

    # Compute denominator
    denominator = 0

    # replace hands to match the specific realization
    fictive_state = state.copy()
    # print(fictive_state)
    for pi in range(game.num_players()):
        for ci in range(game.hand_size()):
            #TODO: HERE IS THE PROBLEM: one can't remove one then add one as potentially one needs more cards then allowed
            # print(pi, ci)
            # print(action.get_deal_move(realisation[pi][ci][0], realisation[pi][ci][1]))
            fictive_state.delete_hand(pi)
    for pi in range(game.num_players()):
        for ci in range(game.hand_size()):
            fictive_state.apply_move(action.get_deal_move(realisation[pi][ci][0], realisation[pi][ci][1]))
            # print(fictive_state)

    # iterate over all possible actions
    #TODO: debug
    # print('Num allowed actions: ', len(fictive_state.legal_moves()))
    for coun, a in enumerate(fictive_state.legal_moves()):
        ns = fictive_state.copy()  # TODO: where should the action be applied? to the actual state or to fictive?
        # this is how different actions makes a difference
        #TODO: debug
        # print(coun)
        # print(a)
        ns.apply_move(a)
        nk = generate_knowledge(game, ns)
        denominator += np.exp(alpha * utility(intention, realisation[plyr_index][card_index], ns, nk))  # TODO: utility currently only takes realisation of a single card into account, but it could now easily work with whole realisation?

    return numerator / denominator
##########################################################################
