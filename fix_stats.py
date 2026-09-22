with open(r'Z:\Desktop\Project\Deep\analytics\statistics.py', 'r') as f:
    content = f.read()

old = '''    # U_less: number of pairs where x < y
    # scipy's alternative="greater" counts x < y pairs
    u_less, _ = stats.mannwhitneyu(x, y, alternative="greater")
    r = 1 - (2 * u_less) / (n_x * n_y)
    return float(r)'''

new = '''    # U_greater: number of pairs where x > y
    # scipy's alternative="less" counts x > y pairs
    u_greater, _ = stats.mannwhitneyu(x, y, alternative="less")
    r = (2 * u_greater) / (n_x * n_y) - 1
    return float(r)'''

if old in content:
    content = content.replace(old, new)
    with open(r'Z:\Desktop\Project\Deep\analytics\statistics.py', 'w') as f:
        f.write(content)
    print('Fixed')
else:
    print('Not found')
    idx = content.find('U_less')
    if idx >= 0:
        print('Found at:', idx)
        print(repr(content[idx:idx+200]))