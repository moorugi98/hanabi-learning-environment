import copy
import numpy as np

# # encode constants here temporally for convenience although ideally it should be encoded in pyhanabi.py
# NUM_PLAYER = game.num_players()
# NUM_HAND = game.hand_size()
COUNTS = [3, 2, 2, 2, 1]
# ALL_COLORS = list(range(game.num_colors()))  # RYGWB
# ALL_RANKS = list(range(game.num_ranks()))
# intentions
PLAY = 0
DISCARD = 1
KEEP = 2
# actions
INVALID = 0
PLAY = 1
DISCARD = 2
REVEAL_COLOR = 3
REVEAL_RANK = 4
DEAL = 5



################## Init & update knowledge ####################

# def initial_knowledge(game):
#     # return max possibility at the start of the game #
#     knowledge = [[[COUNTS[:game.num_ranks()] for col in range(game.num_colors())]
#                   for card in range(game.hand_size())] for plyr in range(game.num_players())]
#     return knowledge
#
#
# def update_knowledge(game, state, knowledge, move):
#     '''
#     update the knowledge after each move
#     : param game: HanabiGame, provides constant information such as num_player
#     :param state: HanabiState, old state, provides information such as cur_player, hands, fireworks
#     :param knowledge: nested list, old knowledge structure that is kept inside game_example.py
#     :param move: HanabiMove, last action  which should be applied to update the knowledge
#     :return: nested list, updated knowledge
#     '''
#
#     def new_draw_knowledge():
#         '''
#         whenever new card is drawn, use this knowledge
#         :return:
#         '''
#         # start with max possibilities
#         card_knowledge = [COUNTS[:game.num_ranks()] for col in range(game.num_colors())]
#         # card already discarded cannot be the new hand
#         for card in state.discard_pile():
#             card_knowledge[card.color()][card.rank()] -= 1
#         # already played
#         for color, stack in enumerate(state.fireworks()):  # R:3, G:2,...
#            for rank in range(stack):
#              card_knowledge[color][rank] -= 1
#         return card_knowledge
#
#     def perfect_identification():
#         for p in range(game.num_players()):
#             for c in range(game.hand_size()):
#                 # only one possibility left means that public agent knows for sure which color and rank it is
#                 if np.sum(np.array(knowledge[p][c]) > 0) == 1:
#                     # get color and rank
#                     col, rank = np.nonzero(np.array(knowledge[p][c]) > 0)
#                     # now every card is 1 instance less likely to be that color and rank combination
#                     for p in range(game.num_players()):
#                         for c in range(game.hand_size()):
#                             new_k[p][c][col[0]][rank[0]] = min(new_k[p][c][col[0]][rank[0]]-1, 0)
#
#     # new knowledge structure to be returned
#     new_k = copy.deepcopy(knowledge)
#
#     if (move.type() == PLAY) or (move.type() == DISCARD):
#         # shift
#         for ci in range(move.card_index(), game.hand_size() - 1):
#             new_k[state.cur_player()][ci] = knowledge[state.cur_player()][ci + 1]
#         # draw new card
#         new_k[state.cur_player()][-1] = new_draw_knowledge()
#
#     elif move.type() == (REVEAL_COLOR) or (REVEAL_RANK):
#         hinted_plyr = (state.cur_player() + move.target_offset()) % game.num_players()
#         print('cur_player, offset: ', state.cur_player(), move.target_offset())
#         if move.type() == REVEAL_COLOR:
#             hands = state.player_hands()[hinted_plyr]
#             for ci, card in enumerate(hands):
#                 # positively identified
#                 if card.color() == move.color():
#                     print('positive: ', ci)
#                     all_color_list = list(range(game.num_colors()))
#                     all_color_list.remove(move.color())
#                     for col in all_color_list:
#                         for rank in range(game.num_ranks()):
#                             # other colors are not possible anymore
#                             new_k[hinted_plyr][ci][col][rank] = 0
#                 # negatively identified
#                 else:
#                     for rank in range(game.num_ranks()):
#                         # color isn't what is just hinted
#                         new_k[hinted_plyr][ci][move.color()][rank] = 0
#
#         elif move.type() == REVEAL_RANK:
#             hands = state.player_hands()[hinted_plyr]
#             for ci, card in enumerate(hands):
#                 if card.rank() == move.rank():
#                     print('positive: ', ci)
#                     all_rank_list = list(range(game.num_ranks()))
#                     all_rank_list.remove(move.rank())
#                     for col in range(game.num_colors()):
#                         for rank in all_rank_list:
#                             new_k[hinted_plyr][ci][col][rank] = 0
#                 else:
#                     for col in range(game.num_colors()):
#                         new_k[hinted_plyr][ci][col][move.rank()] = 0
#
#     else:
#       pass
#
#     # take care of perfectly identified cards
#     perfect_identification()
#
#     return new_k


def generate_knowledge(game, state):
    # start with max possibility
    knowledge = np.array([[[COUNTS[:game.num_ranks()] for col in range(game.num_colors())]
                           for card in range(game.hand_size())] for plyr in range(game.num_players())])

    # discarded
    for card in state.discard_pile():
        knowledge[:, :, card.color(), card.rank()] -= 1

    # already played
    for color, stack in enumerate(state.fireworks()):  # R:3, G:2,...
        for rank in range(stack):
            knowledge[:, :, color, rank] -= 1

    # return HanabiCardKnowledge, which is a public information although accessed using observation
    # which means it doesn't matter which player_index you uses (so fix to 0 for convenience with offset)
    obs = state.observation(0)  # I don't know why, but doing this is one-liner causes weird problems
    hck = obs.card_knowledge()
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

            # only one combination of color,rank is possible. The card is known in other words.
            if np.sum(knowledge[pi, hi] > 0) == 1:
                col, rank = np.nonzero(knowledge[pi, hi] > 0)
                # reduce the possibility for all other cards except the realisation itself
                knowledge[:,:,col[0],rank[0]] = np.maximum(knowledge[:,:,col[0],rank[0]] - 1, 0)
                knowledge[pi,hi,col[0],rank[0]] += 1

                # plyr_index = list(range(game.num_players()))
                # plyr_index.remove(pi)
                # knowledge[plyr_index, :, col[0], rank[0]] = np.maximum(knowledge[plyr_index, :, col[0], rank[0]]-1, 0)

    return knowledge



################################################## INTENTION UPDATE #############################################

def infer_single_joint_intention(game, action, state, knowledge, intention_mat):
    '''
    a method that is called to infer probability of single joint intention instance, e.g. play,discard,discard,keep...
    :param game: HanabiGame, information about constants
    :param action: HanabiMove
    :param state: HanabiState
    :param knowledge: nested list
    :param intention_mat: nested list, 2-dim with 1st dim player and 2nd dim hand of each player
    :return: int, probability of intention vector specified by `intention_mat`
    '''
    prob = 1
    for pi, player in enumerate(intention_mat):
        for i, intention in enumerate(player):
            prob *= pragmatic_listener(game, action, state, knowledge, pi, i)[intention]
    return prob


def get_realisations_probs(game, knowledge, player_index, card_index):
    '''
    returns a list of tuples with the first element being the realisation of a single card (type dictionary)
    and the second element being the probability to get that realisation P(r|c)
    '''

    mylist = []
    for col in range(game.num_colors()):
        for rank in range(game.num_ranks()):
            # realisations that are not possible
            if knowledge[player_index][card_index][col][rank] == 0:
                pass
            else:
                mylist.append(({'color': col, 'rank': rank},
                                knowledge[player_index][card_index][col][rank] / \
                                np.sum(knowledge[player_index][card_index])))
    return mylist


def get_intention_prior():
    # TODO: currently not context dependent
    return 1 / 3


def pragmatic_listener(game, action, state, knowledge, player_index, card_index):
    '''
    return a 3 dim simplex for PLAY,DISCARD,KEEP
    '''

    # 3 dim simplex with prob for each intention
    probs = []

    # compute probability for each intention
    for intention in [PLAY, DISCARD]:
        numerator = 0
        denominator = 0
        # sum over r
        for r, p in get_realisations_probs(game, knowledge, player_index, card_index):
            numerator += pragmatic_speaker(game, action, intention, r, state) * \
                     get_intention_prior() * p
            # sum over i in denom
            for i in [PLAY, DISCARD, KEEP]:
                denominator += pragmatic_speaker(game, action, i, r, state) * \
                             get_intention_prior() * p
        # save the probability
        probs.append(numerator / denominator)

    # P(KEEP|a,c) = 1 - P(PLAY|a,c) - P(DISCARD|a,c)
    probs.append(1 - probs[0] - probs[1])
    return probs


def pragmatic_speaker(game, action, intention, realisation, state):
    '''
    return a scala which is P(action|intention,realisation,context)
    '''
    # TODO: adjust rationality parameter dynamically
    alpha = 1

    # compute numerator
    # TODO: copying doesn't perfectly copy??  state.fireworks()
    new_state = state.copy()
    new_state.apply_move(action)
    new_knowledge = generate_knowledge(game, new_state)
    numerator = np.exp(alpha * utility(intention, realisation, new_state, new_knowledge))



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


def utility(intention, card, state, knowledge):
    '''
    return a utility for a single card of a given realisation
    from various realisations
    '''

    def CardUseless(card, fireworks):
        '''
        return True if the card can be surely discarded
        '''
        if fireworks[card['color']] > int(card['rank']):
            return True
        else:
            return False

    def remaining_copies(card, discard_pile):
        '''
        return number of instance of a given card (if it's relevant) that is still left in the game
        '''
        if card['rank'] == 0:  # rank one
            total_copies = 3
        elif card['rank'] == 4:  # rank five
            total_copies = 1
        else:
            total_copies = 2

        # count how many of the sort given by `card` is discarded
        count = 0
        for discarded in discard_pile:
            col, rank = discarded.color(), discarded.rank()
            if (col == card['color']) and (rank == card['rank']):
                count += 1
        return total_copies - count

    score = 0

    if intention == PLAY:
        # in intention is play and card is playable, this results in one more card on the fireworks. Reward this.
        if state.card_playable_on_fireworks(card['color'], card['rank']):
            score += 10
        # if intention is play and card is not playable at the time
        else:
            # punish loosing a card from stack
            score -= 1
            # and punish loosing a life token
            if state.life_tokens() == 3:
                score -= 1
            elif state.life_tokens() == 2:
                score -= 3
            elif state.life_tokens() == 1:  # game would end directly
                score -= 25

        # if card would still have been relevant in the future,
        # punish loosing it depending on the remaining copies of this card in the deck
        if not CardUseless(card, state.fireworks()):
            if remaining_copies(card, state.discard_pile()) == 2:
                score -= 1
            elif remaining_copies(card, state.discard_pile()) == 1:
                score -= 2
            elif remaining_copies(card, state.discard_pile()) == 0:
                score -= 5


    elif intention == DISCARD:
        # punish loosing a card from stack
        score -= 1
        # reward gaining a hint token:
        score += 0.5
        # punish discarding a playable card
        if state.card_playable_on_fireworks(card['color'], card['rank']):
            score -= 5
        # if card is not playable right now but would have been relevant in the future, punish
        # discarding it depending on the number of remaining copies in the game
        elif not CardUseless(card, state.fireworks()):
            if remaining_copies(card, state.discard_pile()) == 2:
                score -= 1
            elif remaining_copies(card, state.discard_pile()) == 1:
                score -= 2
            elif remaining_copies(card, state.discard_pile()) == 0:
                score -= 5
        # do we want to reward discarding useless card additionally?
        # I think rewarding gaining a hint token should be enough, so nothing happens here
        elif CardUseless(card, state.fireworks()):
            pass


    elif intention == KEEP:
        # keeping a playable card is punished, because it does not help the game
        if CardPlayable(card, state.fireworks()):
            score -= 2
        # if card is not playable right now but is relevant in the future of the game reward keeping
        # this card depending on the remaining copies in the game
        elif not CardUseless(card, state.fireworks()):
            if remaining_copies(card, state.discard_pile()) == 2:
                score += 1
            elif remaining_copies(card, state.discard_pile()) == 1:
                score += 2
            elif remaining_copies(card, state.discard_pile()) == 0:
                score += 5
        # punish keeping a useless card
        elif CardUseless(card, state.fireworks()):
            score -= 1

    return score
##########################################################################