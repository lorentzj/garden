import pandas as pd
import numpy as np
import pickle

ABS_MAX_LENGTH = 100

def create_model_state():
    print("Loading data...")
    name_data = pd.read_csv("data/flora_list.txt", sep = "\t", usecols = ["taxonRank", "scientificName"])
    name_data = name_data[name_data.taxonRank == "SPECIES"].scientificName.str.upper().dropna().tolist()
    print(f"Loaded {len(name_data)} species names")

    print("Creating alphabet...")
    characters = set()
    for name in name_data:
        for c in name[:ABS_MAX_LENGTH]:
            characters.add(c)

    characters = list(characters)
    assert "|" not in characters and "^" not in characters and "$" not in characters
    characters.extend(["^", "$"])

    print("Building subsequences...")
    def gen_n_spans(n):
        for name in name_data:
            name = "^" + name[:ABS_MAX_LENGTH] + "$"
            for i, char in enumerate(name[:-(n + 1)]):
                yield [name[ip] for ip in range(i, i + n + 1)]

    spans_lists = [list(gen_n_spans(i)) for i in range(5)]

    to_prob = lambda l: np.array(l) / sum(l)

    print("Computing transitions...")
    def create_state_dict(spans):
        state_dict = dict()
        for span in spans:
            key = "|".join(span[:-1])
            if key not in state_dict:
                state_dict[key] = np.zeros(len(characters))
            state_dict[key][characters.index(span[-1])] += 1
            
        return {k: to_prob(v) for k, v in state_dict.items()}

    return {"alphabet": characters, "transitions": [create_state_dict(spans_list) for spans_list in spans_lists]}

def predict(transitions, alphabet, prev_chars):
    if len(prev_chars) < len(transitions):
        key = "|".join(prev_chars)
        state_dict  = transitions[len(prev_chars)]
    else:
        key = "|".join(prev_chars[-len(transitions) + 1:])        
        state_dict = transitions[-1]
        
    if key in state_dict:
        return np.random.choice(alphabet, p = state_dict[key])
    else:
        return np.random.choice(alphabet, p = transitions[0][""])

def continuous_predict(transitions, alphabet, max_len):
    generated = ["^"] 
    word_count = 0
    while len(generated) < max_len:
        generated += predict(transitions, alphabet, generated)
        if generated[-1] == "$":
            return "".join(generated[1:-1])
        elif generated[-1] == " ":
            word_count += 1
            if word_count == 2:
                return "".join(generated[1:-1])
    return "".join(generated[1:])

def generate(model_path, max_len):
    with open(model_path, "rb") as handle:
        model = pickle.load(handle)    
    return continuous_predict(model["transitions"], model["alphabet"], max_len)

if __name__ == "__main__":
    save_path = "models/name_gen.pkl"
    model = create_model_state()
    print("Examples:")
    for i in range(10):
        print("\t" + continuous_predict(model["transitions"], model["alphabet"], 100))
    print(f"Model saved to {save_path}")
    with open(save_path, "wb") as handle:
        pickle.dump(model, handle)