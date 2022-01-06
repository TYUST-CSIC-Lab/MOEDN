years=int(input("请输入您要查找的年份：")  )
if((years>=2001) & (years<=2100)):
    if(years%4==0):
        print("是闰年")
    else:
        print("不是闰年")
else:
    print("请输入21世纪的年份")