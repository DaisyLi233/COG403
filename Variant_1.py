import random
from pyClarion import Atom, Atoms, Family, Key
from pyClarion import (
    Agent, Input, ChunkStore, BaseLevel, Pool, Choice, NumDict,
    Event, Priority, Chunk, ks_crawl
)
from datetime import timedelta

# emotional and neutral word banks
emotional_words = [
    "joy", "fear", "anger", "love", "sadness", "hate",
    "pride", "jealousy", "grief", "hope", "disgust", "surprise"
]

neutral_words = [
    "book", "chair", "window", "pencil", "bottle", "computer",
    "table", "paper", "lamp", "phone", "cup", "backpack"
]


# a function to generate the data
def generate_word_list(n_items=12, mixed=True):
    """
    Generate a list of word dictionaries with type labels ('E' or 'N').

    Args:
        n_items (int): Total number of words to generate.
        mixed (bool): If True, generate a mixed list (half emotional, half neutral).
                      If False, generate a list of only emotional words.

    Returns:
        list of dicts: Each dict has 'word' and 'type' keys.
    """
    word_list = []

    if mixed:
        n_each = n_items // 2
        e_words = random.sample(emotional_words, n_each)
        n_words = random.sample(neutral_words, n_each)
        word_list = [{"word": w, "type": "E"} for w in e_words] + \
                    [{"word": w, "type": "N"} for w in n_words]
    else:
        e_words = random.sample(emotional_words, n_items)
        word_list = [{"word": w, "type": "E"} for w in e_words]

    random.shuffle(word_list)
    return word_list


# The data
trial_list = generate_word_list(n_items=12, mixed=True)
print(trial_list)


# KeySpace Definition
class Word(Atoms):
    joy: Atom
    fear: Atom
    anger: Atom
    love: Atom
    sadness: Atom
    hate: Atom
    pride: Atom
    jealousy: Atom
    grief: Atom
    hope: Atom
    disgust: Atom
    surprise: Atom
    book: Atom
    chair: Atom
    window: Atom
    pencil: Atom
    bottle: Atom
    computer: Atom
    table: Atom
    paper: Atom
    lamp: Atom
    phone: Atom
    cup: Atom
    backpack: Atom


class IO(Atoms):
    input: Atom
    output: Atom


class Label(Atoms):
    E: Atom
    N: Atom


class MemoryData(Family):
    word: Word
    io: IO
    label: Label


# Model Construction
d = MemoryData()


class Participant(Agent):
    d: MemoryData
    input: Input
    inhibition: Input
    store: ChunkStore
    blas: BaseLevel
    choice: Choice

    def __init__(self, name: str) -> None:
        p = Family()
        e = Family()
        d = MemoryData()
        super().__init__(name, p=p, e=e, d=d)
        self.d = d

        with self:
            # Input
            self.input = Input("input", (d, d))

            # Memory Store
            self.store = ChunkStore("store", d, d, d)

            # Inhibition
            self.inhibition = Input("inhibition", self.store.chunks, reset=False)

            # Base-level
            self.blas = BaseLevel("blas", p, e, self.store.chunks, sc=1000)

            self.pool = Pool("pool", p, self.store.chunks, func=NumDict.sum)

            # Choice（get one chunk based on the base-level activation）
            self.choice = Choice("choice", p, self.store.chunks)

        # Build the Connection
        self.store.bu.input = self.input.main
        # self.blas.input = self.choice.main
        self.pool["store.bu"] = (
            self.store.bu.main,
            lambda d: d.shift(x=1).scale(x=0.5).logit())
        self.pool["blas"] = (
            self.blas.main,
            lambda d: d.bound_min(x=1e-8).log().with_default(c=0.0))
        self.pool["inhibition"] = self.inhibition.main
        self.choice.input = self.pool.main


        # adding the result "nil"
        self.blas.ignore.add(~self.store.chunks.nil)

    def resolve(self, event: Event) -> None:
        if event.source == self.store.bu.update:
            self.blas.update()
        if event.source == self.blas.update:
            self.choice.trigger()


# Knowledge Initialization
def init_stimuli(d: MemoryData, trial_list: list[dict]) -> list[tuple[str, Chunk]]:
    io, word = d.io, d.word
    stimuli = []

    for item in trial_list:
        w = item["word"]
        chunk = + io.input ** word[w]
        stimuli.append((w, chunk))  # Pack the original word and the chunk (for the mapping purposes)

    return stimuli


# Event Processing
# Initialize agent
agent = Participant("participant")
stimuli = init_stimuli(agent.d, trial_list)

# Compile all the words to ChunkStore
agent.store.compile(*[chunk for _, chunk in stimuli])
agent.breakpoint(timedelta(seconds=1))

# Input all the words to the model
for word_chunk in stimuli:
    agent.system.run_all()
    agent.input.send(word_chunk)  # Input the current word
else:
    agent.system.run_all()

# Build the mapping between chunk key and the word (for converting from chunk key back to the word)
key_to_word = {chunk._name_: word for word, chunk in stimuli}
print("key_to_word:", key_to_word)


# 2. Recall phase: Choice, select a word
recalled_words = []
recalled_keys = set()

# Free recall loop
agent.choice.trigger()
while agent.system.queue:
    event = agent.system.advance()
    if event.source == agent.choice.select:
        response_key = agent.choice.poll()[~agent.store.chunks]
        strength = agent.choice.sample[0][response_key]
        print("the strength of the picked key is:", strength)
        print("the picked key is:", response_key)
        print("Remaining choices:", agent.choice.sample[0])

        if response_key == ~agent.store.chunks.nil:
            break

        recalled_keys.add(response_key)

        chunk_key = str(response_key).split(":")[-1]
        recalled_word = key_to_word.get(chunk_key, None)

        if recalled_word is not None:
            recalled_words.append(recalled_word)

        agent.inhibition.send({response_key: -float("inf")})
        agent.choice.trigger()
        polled = agent.choice.poll()
        print("Choice poll results:", polled)


print("Free recall：", recalled_words)

emo_recalled = [w for w in recalled_words if w in emotional_words]
neu_recalled = [w for w in recalled_words if w in neutral_words]

print(f"Number of emotional words recalled: {len(emo_recalled)}")
print(f"Number of neutral words recalled: {len(neu_recalled)}")
print(f"Total number of words recalled: {len(recalled_words)}")

