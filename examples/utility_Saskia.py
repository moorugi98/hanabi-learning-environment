# intentions
PLAY = 0
DISCARD = 1
KEEP = 2

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
        # in intention is play and card is playable, this results in one more card on the fireworks. Reward this.
        if state.card_playable_on_fireworks(card["color"], card["rank"]):
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
        if state.card_playable_on_fireworks(card["color"], card["rank"]):
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
        if state.card_playable_on_fireworks(card["color"], card["rank"]):
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

