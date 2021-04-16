# intentions
PLAY = 0
DISCARD = 1
KEEP = 2

# first a function from the hanabi learning environment:
def card_playable_on_fireworks(self, color, rank):
    """Returns true if and only if card can be successfully played.
    Args:
    color: 0-based color index of card
    rank: 0-based rank index of card
    """
    return lib.CardPlayableOnFireworks(self._state, color, rank)

######################################
# proper start of the utility function
#####################################

def utility(intention, card, state, knowledge):
    """
    return a utility for a single card of a given realisation
    from various realisations
    """

    def CardUseless(card, fireworks):
        """
        return True if the card can be surely discarded
        """
        if fireworks[card["color"]] > int(card["rank"]):
            return True
        else:
            return False

    def remaining_copies(card, discard_pile):
        """
        return number of instance of a given card (if it's relevant) that is still left in the game
        """
        if card["rank"] == 0:  # rank one
            total_copies = 3
        elif card["rank"] == 4:  # rank five
            total_copies = 1
        else:
            total_copies = 2

        # count how many of the sort given by `card` is discarded
        count = 0
        for discarded in discard_pile:
            col, rank = discarded.color(), discarded.rank()
            if (col == card["color"]) and (rank == card["rank"]):
                count += 1
        return total_copies - count

    score = 0

    if intention == PLAY:

        # [B]: To convey the intention PLAY, there has most likely been given a
        # hint, which means losing an information token. Punish this mildly.
        # Punish it a bit more when there are only 1-2 hints left.
        if state.information_tokens() > 2:
            score += -0.5
        elif state.information_tokens() > 0:
            score += -1
        # [B]: Punsish if the agent after the playing agent will not have an
        # information token available anymore (which might result in criticial
        # cards being discarded)
        elif state.information_tokens() == 0:
                score += -3

        # [B, only added comment]: case 1: card is playable
        # if intention is play and card is playable, this results in one more card on
        # the fireworks. Reward this.
        card_color = card["color"]
        card_rank = card["rank"]
        if state.card_playable_on_fireworks(card_color, card_rank):
            # [B]: check if the same card is already cued as 'playable'
            # on some other player's hand first
            # if ((card_on_hands['color'] is card_color) and (card_on_hands['rank'] is rank_color)):
            #     # [B]: #TODO: check if other card instance is already hinted as "PLAY" or "KEEP"
            #     # if YES:
            #     #   score += -1
            if True:
                score += 10

        # [B, only added comment]: case 2: card is not playable right now
        else:
            # punish loosing a card from stack
            score += -1
            # and punish loosing a life token
            if state.life_tokens() == 3:
                score += -1
            elif state.life_tokens() == 2:
                score += -3
            elif state.life_tokens() == 1:  # game would end directly
                score += -25

        # if card would still have been relevant in the future,
        # punish loosing it depending on the remaining copies of this card in the deck
        if not CardUseless(card, state.fireworks()):
            if remaining_copies(card, state.discard_pile()) == 2:
                score += -1
            elif remaining_copies(card, state.discard_pile()) == 1:
                score += -5 # [B]: upped value from 2 to 5
            # [B, only added comment]: Case 3: card is critical
            elif remaining_copies(card, state.discard_pile()) == 0:
                # [B]: case differentiation:
                if card["rank"] == 5:
                    score += -8
                elif card["rank"] == 4:
                    score += -10
                # [B]: losing a critical 1,2,3 would reduce the total
                # game score the most, punish it accordingly.
                else:
                    score += -15

    elif intention == DISCARD:
        # punish loosing a card from stack
        # [B]: changed value from -1 to -0.5
        score += -0.5
        # reward gaining a hint token:
        # [B]: changed value from +0.5 to +1
        score += 1

        # punish discarding a playable card
        if state.card_playable_on_fireworks(card["color"], card["rank"]):
            score += -2
            # [B]: playable 2,3 advance the game more than 4,5
            # punish losing a playable 2,3 more (we start indexing at 0)
            if card["rank"] == 2 or card["rank"] == 3:
                score += -3

        # if card is not playable right now but would have been relevant in the future, punish
        # discarding it depending on the number of remaining copies in the game
        elif not CardUseless(card, state.fireworks()):
            card_color = card["color"]
            card_rank = card["rank"]
            # [B]: Check if to be discarded card is already clued on someone's
            # hand, otherwise punish that -0.5
            # # (TODO: estimated_hands in deepmind framework?)
            # for card_on_hands in state.estimated_hands[:]:
            #     if ((card_on_hands['color'] is card_color) and (card_on_hands['rank'] is rank_color)):
            #         # [B]: #TODO: check if other card instance is already hinted as "PLAY" or KEEP
            #         # if YES:
            #         #   score += 1
            #     else:
            #         score += -0.5

            # [B, only added comment]: How bad is discarding the
            # particular card for the game /end score?
            if remaining_copies(card, state.discard_pile()) == 2:
                score += -1
            elif remaining_copies(card, state.discard_pile()) == 1:
                score += -5 # [B]: upped value from 2 to 5
            # [B, only added comment]: Case 3: card is critical
            elif remaining_copies(card, state.discard_pile()) == 0:
                # [B]: case differentiation:
                if card["rank"] == 5:
                    score += -8
                elif card["rank"] == 4:
                    score += -10
                # [B]: losing a critical 1,2,3 would reduce the total
                # game score the most, punish it accordingly.
                else:
                    score += -15

        # do we want to reward discarding useless card additionally?
        # I think rewarding gaining a hint token should be enough, so nothing happens here
        elif CardUseless(card, state.fireworks()):
            pass

    elif intention == KEEP:
        # [B, only added comment]: Case 1: card is playable right now
        # keeping a playable card is punished, because it does not help the game
        # [B]: I'm unsure about this part
        if state.card_playable_on_fireworks(card['color'], card['rank']):
            score += -2

        # [B]: Case 2: card will be playable soon (with one additional card needed on the stack)
        elif state.card_playable_on_fireworks(card["color"], card["rank"] + 1):
            score += 3
            # [B]: on top of that, check if this additional card is already on
            # someone's hand with the intention of PLAY (by acessing the public
            # vector of intention and the public knowledge vector), if yes:
                # score += 2

        # Check if card is criticial
        elif not CardUseless(card, state.fireworks()):
            if remaining_copies(card, state.discard_pile()) == 2:
                score += 1
            elif remaining_copies(card, state.discard_pile()) == 1:
                score += 4 # [B]: upped value from 2 to 4
            # [B, only added comment]: Case 3: card is critical
            elif remaining_copies(card, state.discard_pile()) == 0:
                # [B]: Case Differentiation:
                if card["rank"] == 5:
                    score += 5
                elif card["rank"] == 4:
                    score += 8
                # [B]: saving a critical 1,2,3 will be most beneficial
                # to the end score.
                else:
                    score += 15

                # [B]: Test if card is oldest / in the oldest slot on hand
                # Bonus for avoiding a critical card being the oldest and most
                # likely to be discarded (score += 2)
                # TODO: Is that currently included in our representation?

        # [B, only added comment]: Case 4: card is useless
        elif CardUseless(card, state.fireworks()):
            #[B, only added comment]: punishment for saving an irrelevant card
            score += -2 # [B]: upped value from 1 to 2

        # [B]: to convey the intention of KEEP, there have most probably
        # been given a hint, which means losing an information token.
        # Punish this mildly. Punish it a bit more when there are only 1-2 hints left.
        if state.information_tokens() > 2:
            score += -0.5
        elif (state.information_tokens() > 0):
            score += -1
        # [B]: Punsish if the agent after the playing agent will not have an
        # information token available anymore (which might result in criticial
        # cards being discarded)
        elif state.information_tokens() == 0:
            score += -3

        ######################################
        #TODO:

        # [B]: from Georg's example: Next round, the person will either play on
        # old clue (+0.2 if succesfull) or they will discard (-0.2)
        # TODO: test if there is a card with the intention PLAY already on the
        # hand of the player, if not: -0.2 (by acessing the public intention vector)
        # TODO: if there is a card with intention PLAY: check if it is indeed
        # playable?

        # [B]: penalty for blocking a slot on the hand of the player
        # TODO: check how many slots are currently blocked with card that
        # have the intention KEEP (by acessing the public intention vector)
        ######################################

    return score

# [B]:
# Observations:
# computing utility scores per card makes it hard to reward hints that
# are concerning 2 cards (and are beneficial for one card and maybe neutral for
# the other). It may be somehow included in the multiplication of the scores though.

# Punish if something else is more relveant right now (other hint giving for example)
# How?

# check if next person should be prevented from discarding their oldest card
# by using the public knowledge vector only (so we might not have enough information)
# to decided that, then make the action/ intention of playing less likely
