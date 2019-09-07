def crop(t):
    for i in range(np.shape(t)[0]):
        if False in (t[i,:]==0):
            up=i
            break
        if i == np.shape(t)[0]-1:
            return(t)
    for i in reversed(range(np.shape(t)[0])):
        if False in (t[i,:]==0):
            down=i
            break
    for i in range(np.shape(t)[0]):
        if False in (t[:,i]==0):
            left=i
            break
    for i in reversed(range(np.shape(t)[0])):
        if False in (t[:,i]==0):
            right=i
            break

    wid = right-left
    hei = down - up
    center_x = (left+right)//2
    center_y = (up+down)//2

    if wid > hei:
        half_len = wid//2
    else:
        half_len = hei//2
    a = center_y-half_len # up
    b = center_y+half_len # down
    c = center_x-half_len # left
    d = center_x+half_len # right
    if a <0:
        a = 0
    if c<0:
        c=0
    return(t[a:b, c:d])