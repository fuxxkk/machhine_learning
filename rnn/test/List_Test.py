import pickle
# save_file = open("/Applications/pywork/machhine_learning/rnn/test/list_file", "wb+")
load_file = open("/Applications/pywork/machhine_learning/rnn/test/list_file", "rb+")
l = list()
l.append("a")
print(l)


# def save_list(file):
#     pickle.dump(l, file)
    # save_file.write(data)
    # save_file.close()


# save_list(save_file)

def load(file):
    ll=pickle.load(file)
    print(ll)
    print(type(ll))

load(load_file)