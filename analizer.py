import numpy as np
import mpmath as mp
import skimage.measure
from PIL import Image
from scipy.special import comb

import math
import pprint

from interface_data import DATA 
mp.dps = 50
def calc_laconic(info_groups, rgb_entropy):
    # hardcode becouse of difficulty
    # see https://stats.stackexchange.com/questions/207893/entropy-of-the-multinomial-distribution
    group = info_groups['info_element_group1']
    n = group['elements_count']
    p = group['states_probs']
    res = -math.log2(math.factorial(n))
    # todo: replace with MP
    res -= n * np.sum([p_i * math.log2(p_i) for p_i in p])
    for p_i in p:
        for x in range(n+1):
            res += comb(n, x) * (p_i ** x) * ((1-p_i)**(n-x)) * math.log2(math.factorial(x))
    r1 = 0
    for i in range(n+1):
        for k in range(n+1-i):
            p1 = mp.exp(mp.log(p[0])*i + mp.log(p[1])*k + mp.log(p[2])*(n-i-k))
            print(p1)
            r1 -= comb(n, i)*comb(n-i, k)*p1*mp.log(p1,2)
    return float(res/r1)
    
def calc_simplicity(inputs, rgb_entropy):
    return 1./(rgb_entropy*inputs["groups_count"]*inputs["inputs_count"])

def calc_criterias(interface):
    img = Image.open(interface['screenshot'])
    rgb_entropy = skimage.measure.shannon_entropy(img)
    laconic = calc_laconic(interface['info_groups'], rgb_entropy)
    simplicity = calc_simplicity(interface['inputs'], rgb_entropy)
    
    return {
        "Простота": simplicity,
        "Лаконичность": laconic,
        "RGB энтропия": rgb_entropy
    }

def main():
    for interface in DATA:
        pprint.pprint(calc_criterias(interface))

if __name__ == "__main__":
    main()