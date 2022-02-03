
train2=train.copy()
train2.reset_index(inplace=True)

def get_month(x):
    if x=='Ene':
        y=1
    elif x=='Feb':
        y=2
    elif x=='Mar':
        y=3
    elif x=='Abr':
        y=4
    elif x=='May':
        y=5
    elif x=='Jun':
        y=6
    elif x=='Jul':
        y=7
    elif x=='Ago':
        y=8
    elif x=='Sep':
        y=9
    elif x=='Oct':
        y=10
    elif x=='Nov':
        y=11
    elif x=='Dic':
        y=12
    else:
        y=0
    return y


train2['month']=train2['index'].str[:3]
train2['month']=train2.apply(lambda row: get_month(row['month']),axis=1)
