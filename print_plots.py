import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def scatter_plot(pos_sim, model_dir):
    positions = {0: [], 1: [], 2: [], 3: [], 4: [], -1: []}
    all_count = len(pos_sim)
    for item in pos_sim:
        positions[item['pos']].append(item['sim'])

    missed_count = len(positions[-1])
    found_count = all_count - missed_count

    print([(x, len(positions[x])) for x in positions])

    fig, ax = plt.subplots()
    red_patch = mpatches.Patch(color='salmon', label='Missed', alpha=0.50)
    green_patch = mpatches.Patch(color='limegreen', label='Identified', alpha=0.50)
    ax.legend(handles=[red_patch, green_patch])

    plt.scatter(range(len(positions[0])), positions[0], color='limegreen', alpha=0.50, marker='.')
    plt.scatter(range(len(positions[1])), positions[1], color='limegreen', alpha=0.50, marker='^')
    plt.scatter(range(len(positions[2])), positions[2], color='limegreen', alpha=0.60, marker='s')
    plt.scatter(range(len(positions[3])), positions[3], color='limegreen', alpha=0.60, marker='p')
    plt.scatter(range(len(positions[4])), positions[4], color='limegreen', alpha=0.50, marker='h')
    plt.scatter(range(len(positions[-1])), positions[-1], color='salmon', alpha=0.70, marker='2')

    plt.title('Similarity Distribution, Total:{}, Identified:{}'.format(all_count, found_count))
    plt.savefig(model_dir + '/scatter.png')

    plt.show()
    print('Done')


def histo_sim(pos_sim, model_dir):
    similarities = []
    for item in pos_sim:
        if item['pos'] >= 0:
            similarities.append(item['sim'])

    fig, ax = plt.subplots()
    ax.set_title('Similarity Distribution')
    ax.set_xlabel('Similarity Percentage ->')
    ax.set_ylabel('Bug Reports Count->')
    plt.hist(similarities)
    plt.savefig(model_dir + '/histo_sim.png')
    plt.show()
    print('Done')


def top5_dist(pos_sim, model_dir):
    positions = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, -1: 0}
    for item in pos_sim:
        if item['pos'] >= 0:
            positions[item['pos']] += 1

    fig, ax = plt.subplots()
    ax.set_title('Top 5 Distribution')
    ax.set_xlabel('Position ->')
    ax.set_ylabel('Bug Reports Count ->')
    langs = [1, 2, 3, 4, 5]
    students = list(positions.values())[:5]
    ax.bar(langs, students)
    plt.savefig(model_dir + '/top5_dist.png')
    plt.show()


# if __name__ == '__main__':
#     all_data = []
#     print(len(all_data))
#     scatter_plot(all_data)
#     top5_dist(all_data)
#     histo_sim(all_data)
