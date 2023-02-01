
import numpy as np

data = np.load('./all_speed_data.npz')
speed_data, mask = data['arr_0'], data['arr_1']

# print(speed_data.shape)
# print(mask.shape)
# (14, 288, 4248, 1)
# (14, 288, 4248, 1)

speed_data, mask = np.squeeze(speed_data), np.squeeze(mask)

gaofeng = np.concatenate([speed_data[:,84:120,:],speed_data[:,204:240,:]],axis=1)
# print(gaofeng.shape)
not_gaofeng = np.concatenate([speed_data[:,:84,:],speed_data[:,120:204,:],speed_data[:,240:,:]], axis=1)
# print(not_gaofeng.shape)

gaofeng_mask = np.concatenate([mask[:,84:120,:],mask[:,204:240,:]],axis=1)
not_gaofeng_mask = np.concatenate([mask[:,:84,:],mask[:,120:204,:],mask[:,240:,:]], axis=1)

weekday = np.concatenate([speed_data[0:1,...], speed_data[3:8, ...], speed_data[10:14,...]], axis=0)
weekend = np.concatenate([speed_data[1:3, ...], speed_data[8:10, ...]], axis=0)
print(weekend.shape,weekday.shape)
weekday_mask = np.concatenate([mask[0:1,...], mask[3:8, ...], mask[10:14, ...]], axis=0)
weekend_mask = np.concatenate([mask[1:3, ...], mask[8:10, ...]], axis=0)

weekend, weekend_mask = np.reshape(weekend,[-1,4248]), np.reshape(weekend_mask, [-1, 4248])
weekday, weekday_mask = np.reshape(weekday, [-1, 4248]), np.reshape(weekday_mask, [-1, 4248])

speed_data = np.reshape(speed_data, [-1, 4248])
sum_ = np.sum(speed_data, axis=0)
count_nonzero = np.count_nonzero(speed_data, axis=0)
mean = np.divide(sum_, count_nonzero)
# print(sum_.shape,count_nonzero.shape,len(mean))     # (4248,) (4248,) 4248
# print(np.sum(np.isnan(mean)))   # 结果有8个是nan，

gaofeng = np.reshape(gaofeng, [-1, 4248])
not_gaofeng = np.reshape(not_gaofeng, [-1, 4248])
gaofeng_mask = np.reshape(gaofeng_mask, [-1, 4248])
not_gaofeng_mask = np.reshape(not_gaofeng_mask, [-1, 4248])


def compute_var(data, mask, mean):
    tmp = []

    for i in range(data.shape[-1]):
        if np.isnan(mean[i]):
            continue
        td = data[:, i]
        tm = mask[:, i]
        after_mask = td[tm]
        # print(len(after_mask))
        tmp.append(np.divide(np.sum(list(map(lambda x: np.square(x-mean[i]), after_mask))), len(after_mask)))
    return tmp

def compute_var2(data, mask):
    tmp = []

    for i in range(data.shape[-1]):

        td = data[:, i]
        tm = mask[:, i]
        after_mask = td[tm]
        if len(after_mask)==0:

            print(len(after_mask))
        else:
            tmp.append(np.var(after_mask))
    return tmp

# t = compute_var( gaofeng, gaofeng_mask, mean)
# print(t)
# print(len(t))  # 4240
# tt = compute_var(not_gaofeng, not_gaofeng_mask, mean)
# print(tt)   # 4240
# print(len(tt))
# print(len(np.array(t)>np.array(tt)))

t = compute_var2( gaofeng, gaofeng_mask)
print(t)
print(len(t))  # 4240
tt = compute_var2(not_gaofeng, not_gaofeng_mask)
print(tt)   # 4240
print(len(tt))
print(len(np.array(t)>np.array(tt)))


# print(weekday.shape,weekend.shape)
# print(weekday_mask.shape, weekend_mask.shape)
# ttt = compute_var(weekday, weekday_mask, mean)
# print(ttt)
# print(len(ttt))  # 4240
# tttt = compute_var(weekend, weekend_mask, mean)
# print(tttt)   # 4240
# print(len(tttt))
# print(len(np.array(ttt)>np.array(tttt)))

# print(weekday.shape,weekend.shape)
# print(weekday_mask.shape, weekend_mask.shape)
# ttt = compute_var2(weekday, weekday_mask)
# print(ttt)
# print(len(ttt))  # 4240
# tttt = compute_var2(weekend, weekend_mask)
# print(tttt)   # 4240
# print(len(tttt))
# print(len(np.array(ttt)>np.array(tttt)))