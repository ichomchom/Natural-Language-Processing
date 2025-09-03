with open ("./nyt.txt", "r") as file:
    lines = file.readlines()
    d = {}
    for line in lines:
        words = line.split()
        for word in words:
            word = word.strip('.,!?";:-()[]{}').lower()
            if word in d:
                d[word] += 1
            else:
                d[word] = 1
    sorted_d = dict(sorted(d.items(), key=lambda item: item[1], reverse=True))
    top_10 = {k: v for i, (k, v) in enumerate(sorted_d.items()) if i < 10}

print(top_10)