

if __name__ == "__main__":
    ft_unlabel = open('gene.train.unlabel','w')
    with open('gene.train','r') as f:
        for line in f:
            line_list = line.strip().split(" ")
            if not line_list:
                ft_unlabel.write('\n')
            else:
                ft_unlabel.write(line_list[0] + '\n')
    ft_unlabel.close()
