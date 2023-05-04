import math
import sys
import matplotlib.pyplot as plt

from timeit import default_timer as timer
import random as rng
import datetime as dt
import pandas as pd
import string
from numba import jit,cuda
import numpy as np

sys.setrecursionlimit(600000)

MIN_MERGE = 32
sortsAmount=10


@jit(target_backend='cuda')
def GnomeSort(a): #Worst case: n^2, best case: n, average: n^2
    n=len(a)
    index =0
    iter=0
    while index<n:
        iter+=1
        if index==0:
            index+=1
        if a[index]>=a[index-1]:
            index+=1
        else:
            a[index],a[index-1]=a[index-1],a[index]
            index-=1
    return iter

@jit(target_backend='cuda')
def BubbleSort(a): #Worst case: n^2, best case: n, average: n^2
    iter = 0
    swapped = False
    for i in range(1, len(a)):
        for j in range(0, len(a) - i):
            iter += 1
            if a[j] > a[j + 1]:
                swapped = True
                a[j], a[j + 1] = a[j + 1], a[j]
        if not swapped:
            return iter
    return iter

@jit(target_backend='cuda')
def InsertionSort(a): #Worst case: n^2 Best case: n Average: n^2
    iter = 0
    for i in range(1, len(a)):
        key = a[i]
        j = i - 1
        while j >= 0 and key < a[j]:
            iter += 1
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = key
    return iter


@jit(target_backend='cuda')
def SelectionSort(a): #Worst case: n^2 Best case: n Average: n^2
    iter = 0
    for i in range(len(a)):
        minInd = i
        for j in range(i + 1, len(a)):
            iter += 1
            if a[j] < a[minInd]:
                minInd = j
        a[i], a[minInd] = a[minInd], a[i]
    return iter


@jit(target_backend='cuda')
def CocktailSort(a): #Worst case: n^2 Best case: n Average: n^2
    iter = 0
    swapped = True
    l = 0
    r = len(a) - 1
    while (swapped == True):
        for i in range(l, r):
            iter += 1
            if a[i] > a[i + 1]:
                a[i], a[i + 1] = a[i + 1], a[i]
                swapped = True
        if not swapped:
            break
        swapped = False
        r = r - 1
        for i in range(r - 1, l - 1, -1):
            iter += 1
            if a[i] > a[i + 1]:
                a[i], a[i + 1] = a[i + 1], a[i]
                swapped = True
        l = l + 1
    return iter


@jit(target_backend='cuda')
def Partition(a,low,high):
    pivot=a[high]
    i=low-1
    for j in range(low,high):
        if a[j]<=pivot:
            i=i+1
            a[i],a[j]=a[j],a[i]
    a[i+1],a[high]=a[high],a[i+1]
    return i+1

@jit(target_backend='cuda')
def PartitionIterCount(a,low,high):
    pivot=a[high]
    iter=0
    i=low-1
    for j in range(low,high):
        iter+=1
        if a[j]<=pivot:
            i=i+1
            a[i],a[j]=a[j],a[i]
    a[i+1],a[high]=a[high],a[i+1]
    return iter

@jit(target_backend='cuda')
def QuickSort(a,low,high): #Worst case: n^2 Best case: n*log(n) Average: n*log(n)
    iter = 0
    if low < high:
        iter += 1
        iter += PartitionIterCount(a,low,high)
        pi = Partition(a, low, high)
        iter += QuickSort(a, low, pi - 1)
        iter += QuickSort(a, pi + 1, high)
    return iter

@jit(target_backend='cuda')
def Merge(a,l,m,r):
    iter = 0
    n1 = m - l + 1
    n2 = r - m

    L = [0] * (n1)
    R = [0] * (n2)

    for i in range(0, n1):
        iter+=1
        L[i] = a[l + i]
    for j in range(0, n2):
        iter+=1
        R[j] = a[m + 1 + j]

    i = 0
    j = 0
    k = l

    while i < n1 and j < n2:
        if L[i] <= R[j]:
            iter += 1
            a[k] = L[i]
            i += 1
        else:
            iter += 1
            a[k] = R[j]
            j += 1
        k += 1

    while i < n1:
        iter += 1
        a[k] = L[i]
        i += 1
        k += 1

    while j < n2:
        iter += 1
        a[k] = R[j]
        j += 1
        k += 1
    return iter


@jit(target_backend='cuda')
def MergeSort(a,l,r): #Worst case: O(n*log(n)) Best case: Omega(n*log(n)) Average: Theta(n*log(n))
    iter = 0
    if l < r:
        iter += 1
        m = l + (r - l) // 2

        iter += MergeSort(a, l, m)
        iter += MergeSort(a, m + 1, r)
        iter += Merge(a, l, m, r)
    return iter

@jit(target_backend='cuda')
def Heapify(a,n,i):
    iter = 0
    largest = i
    l = 2 * i + 1
    r = 2 * i + 2

    if l < n and a[i] < a[l]:
        iter += 1
        largest = l
    if r < n and a[largest] < a[r]:
        iter += 1
        largest = r
    if largest != i:
        iter += 1
        a[i], a[largest] = a[largest], a[i]
        iter += Heapify(a, n, largest)
    return iter


@jit(target_backend='cuda')
def HeapSort(a): #Worst case: n*log(n) Best case: n*log(n) or n (equal keys) Average: n*log(n)
    iter = 0
    n = len(a)
    for i in range(n // 2 - 1, -1, -1):
        iter += Heapify(a, n, i)
    for i in range(n - 1, 0, -1):
        a[i], a[0] = a[0], a[i]
        iter += Heapify(a, i, 0)
    return iter

@jit(target_backend='cuda')
def CountingSort(array, place):
    iter = 0
    size = len(array)
    output = [0] * size
    count = [0] * 10

    for i in range(0, size):
        iter += 1
        index = array[i] // place
        count[index % 10] += 1

    for i in range(1, 10):
        iter += 1
        count[i] += count[i - 1]

    i = size - 1
    while i >= 0:
        iter += 1
        index = array[i] // place
        output[count[index % 10] - 1] = array[i]
        count[index % 10] -= 1
        i -= 1

    for i in range(0, size):
        array[i] = output[i]
    return iter

@jit(target_backend='cuda')
def RadixSort(a):
    iter = 0
    if type(a[0]) != type(10):
        print("Can't be used with this data type")
        return
    max_element = max(a)

    place = 1
    while max_element // place > 0:
        iter += CountingSort(a, place)
        place *= 10
    return iter

@jit(target_backend='cuda')
def CalculateMinRun(n):
    r=0
    while n>=MIN_MERGE:
        r|= n&1
        n>>=1
    return n+r

@jit(target_backend='cuda')
def LRInsertionSort(a,l,r):
    iter = 0
    for i in range(l + 1, r + 1):
        j = i
        while j > l and a[j] < a[j - 1]:
            iter += 1
            a[j], a[j - 1] = a[j - 1], a[j]
            j -= 1
    return iter

@jit(target_backend='cuda')
def TimSort(a): #Worst case: n*log(n) Best case: n Average: n*log(n)
    iter = 0
    n = len(a)
    minRun = CalculateMinRun(n)

    for start in range(0, n, minRun):
        iter+=1
        end = min(start + minRun - 1, n - 1)
        iter += LRInsertionSort(a, start, end)

    size = minRun
    while size < n:
        for left in range(0, n, 2 * size):
            mid = min(n - 1, left + size - 1)
            right = min((left + 2 * size - 1), (n - 1))
            iter+=1
            if mid < right:
                iter += Merge(a, left, mid, right)
        size *= 2
    return iter

numArrays= [[0]*50,[0]*500,[0]*5000,[0]*50000,[0]*50,[0]*500,[0]*5000,[0]*50000,[0]*50,[0]*500,[0]*5000,[0]*50000]
intArrays= [[0]*50,[0]*500,[0]*5000,[0]*50000,[0]*50,[0]*500,[0]*5000,[0]*50000,[0]*50,[0]*500,[0]*5000,[0]*50000]
strArrays= [['']*50,['']*500,['']*5000,['']*50000,['']*50,['']*500,['']*5000,['']*50000,['']*50,['']*500,['']*5000,['']*50000]
dateArrays=[[dt.date.min]*50,[dt.date.min]*500,[dt.date.min]*5000,[dt.date.min]*50000,[dt.date.min]*50,[dt.date.min]*500,[dt.date.min]*5000,[dt.date.min]*50000,[dt.date.min]*50,[dt.date.min]*500,[dt.date.min]*5000,[dt.date.min]*50000]
for i in range(4): #Fully random
    for j in range (len(intArrays[i])):
        numArrays[i][j]=rng.randint(0,9)
        intArrays[i][j]=rng.randint(1, 1000)
        strArrays[i][j]=''.join(rng.choices(string.ascii_letters,k=rng.randint(1,100)))
        dateArrays[i][j]=dt.date(rng.randint(1,9999),rng.randint(1,12),rng.randint(1,28))
for i in range(4,8): #Half sorted, half random (except nums, they repeat)

    for j in range(len(intArrays[i])//2):
        numArrays[i][j]=j%10
        intArrays[i][j]=j
        strArrays[i][j]=str(j)
        if j!=0:
            dateArrays[i][j]=dateArrays[i][j-1]+dt.timedelta(days=1)
    for j in range(len(intArrays[i])//2,len(intArrays[i])):
        numArrays[i][j] = rng.randint(0, 9)
        intArrays[i][j] = rng.randint(1, 1000)
        strArrays[i][j] = ''.join(rng.choices(string.ascii_letters, k=rng.randint(1, 100)))
        dateArrays[i][j] = dt.date(rng.randint(1, 9999), rng.randint(1, 12), rng.randint(1, 28))
for i in range(8,12): #Mostly (~80%) sorted, rest random (except nums, they repeat)
    for j in range(len(intArrays[i])*8//10):
        numArrays[i][j]=j%10
        intArrays[i][j]=j
        strArrays[i][j]=str(j)
        if j!=0:
            dateArrays[i][j]=dateArrays[i][j-1]+dt.timedelta(days=1)
    for j in range(len(intArrays[i])*8//10,len(intArrays[i])):
        numArrays[i][j] = rng.randint(0, 9)
        intArrays[i][j] = rng.randint(1, 1000)
        strArrays[i][j] = ''.join(rng.choices(string.ascii_letters, k=rng.randint(1, 100)))
        dateArrays[i][j] = dt.date(rng.randint(1, 9999), rng.randint(1, 12), rng.randint(1, 28))


SortTypes= [GnomeSort,BubbleSort,InsertionSort,SelectionSort,CocktailSort,QuickSort,MergeSort,HeapSort,RadixSort,TimSort]
SortingTimes=np.zeros((sortsAmount,48),dtype=float)
SortingIters=np.zeros((sortsAmount,48),dtype=float)
AverageIters=np.zeros((sortsAmount,48),dtype=float)

for i in range(10):
    if i<5:
        for j in range(48):
            AverageIters[i][j] = len(intArrays[j%12])*len(intArrays[j%12])
    if i >= 5 and i != 8:
        for j in range(48):
            AverageIters[i][j] = len(intArrays[j%12])*math.log(len(intArrays[j%12]))


dfAverage=pd.DataFrame(AverageIters)


for i in range(sortsAmount):
    if i!=5 and i!=6:
        for j in range(12):
            start= timer()
            SortingIters[i][j]=SortTypes[i](numArrays[j%12].copy())
            SortingTimes[i][j]=timer()-start
            print('Done!')
        for j in range(12,24):
            start= timer()
            SortingIters[i][j]=SortTypes[i](intArrays[j%12].copy())
            SortingTimes[i][j]=timer()-start
            print('Done!')
        if i !=8:
            for j in range(24,36):
                start= timer()
                SortingIters[i][j]=SortTypes[i](strArrays[j%12].copy())
                SortingTimes[i][j]=timer()-start
                print('Done!')
            for j in range(36,48):
                start= timer()
                SortingIters[i][j]=SortTypes[i](dateArrays[j%12].copy())
                SortingTimes[i][j]=timer()-start
                print('Done!')
    else:
        for j in range(12):
            start= timer()
            SortingIters[i][j]=SortTypes[i](numArrays[j%12].copy(),0,len(numArrays[j%12])-1)
            SortingTimes[i][j]=timer()-start
            print('Done!')
        for j in range(12,24):
            start= timer()
            SortingIters[i][j]=SortTypes[i](intArrays[j%12].copy(),0,len(intArrays[j%12])-1)
            SortingTimes[i][j]=timer()-start
            print('Done!')
        if i !=8:
            for j in range(24,36):
                start= timer()
                SortingIters[i][j]=SortTypes[i](strArrays[j%12].copy(),0,len(strArrays[j%12])-1)
                SortingTimes[i][j]=timer()-start
                print('Done!')
            for j in range(36,48):
                start= timer()
                SortingIters[i][j]=SortTypes[i](dateArrays[j%12].copy(),0,len(dateArrays[j%12])-1)
                SortingTimes[i][j]=timer()-start
                print('Done!')

dfTimes=pd.DataFrame(SortingTimes)
dfIters=pd.DataFrame(SortingIters)

dfAverage.to_excel(r'C:\Users\grigo\Documents\Averages.xlsx')
dfIters.to_excel(r'C:\Users\grigo\Documents\Iters.xlsx')
dfTimes.to_excel(r'C:\Users\grigo\Documents\Time.xlsx')