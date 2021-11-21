#!/usr/bin/env python3

import os
import sys
import numpy as np
import netCDF4 as nc
import glob

site = sys.argv[1]

outFile = open(f'/nesi/nobackup/vuw03073/bigboy/pdfs/{site}_alltraj_flat.txt', 'w')
xs = np.array(())
ys = np.array(())
for file in sorted(glob.glob(f'/nesi/nobackup/vuw03073/bigboy/all_settlement/{site}*')):
	traj = nc.Dataset(file)
	print(file)
	lon = traj.variables['lon'][:]
	lat = traj.variables['lat'][:]
	x = lon[np.where(lon.mask==False)]
	y = lat[np.where(lat.mask==False)]
	xs = np.append(xs,x)
	ys = np.append(ys,y)
pts = np.array((xs,ys))
np.savetxt(outFile, pts)
outFile.close()

pdf_OPO = np.zeros((801,761))
pdf_MAU = np.zeros((801,761))
pdf_WEST = np.zeros((801,761))
pdf_FLE = np.zeros((801,761))
pdf_TAS = np.zeros((801,761))
pdf_LWR = np.zeros((801,761))
pdf_CAP = np.zeros((801,761))
pdf_CAM = np.zeros((801,761))
pdf_KAI = np.zeros((801,761))
pdf_GOB = np.zeros((801,761))
pdf_TIM = np.zeros((801,761))
pdf_HSB = np.zeros((801,761))
pdf_BGB = np.zeros((801,761))
pdf_FIO = np.zeros((801,761))


pdfs = [pdf_OPO,pdf_MAU,pdf_WEST,pdf_FLE,pdf_TAS,pdf_LWR,pdf_CAP,pdf_CAM,pdf_KAI,pdf_GOB,pdf_TIM,pdf_HSB,pdf_BGB,pdf_FIO]
sites = ['OPO','MAU','WEST','FLE','TAS','LWR','CAP','CAM','KAI','GOB','TIM','HSB','BGB','FIO']

lon_edges = np.linspace(165, 184, 381)
lat_edges = np.linspace(-52, -32, 401)
lon_edges = np.linspace(165, 184, 381*2)
lat_edges = np.linspace(-52, -32, 401*2)

for i in range(len(sites)):
for file in sorted(glob.glob(f'/nesi/nobackup/vuw03073/bigboy/all_settlement/GOB*')):
	print(file)
	ym = file[-9:-3]
	traj = nc.Dataset(file)
	lon = traj.variables['lon'][:]
	lat = traj.variables['lat'][:]
	x = lon[np.where(lon.mask==False)]
	x[x<0] += 360
	y = lat[np.where(lat.mask==False)]
	H, _, _ = np.histogram2d(y, x, [lat_edges, lon_edges])#, density = True)
	#pdfs[i] += H
	outFile = open(f'bigboy_pdf/oneeighty/KAI_GOB/GOB_{ym}_pdf.txt', 'w')
	np.savetxt(outFile, H)
	outFile.close()

outFile = open(f'FIO_pdf.txt', 'w')
np.savetxt(outFile, pdf)
outFile.close()

opo = np.loadtxt('OPO_pdf.txt')


pdf_OPO = np.loadtxt('bigboy_pdf/OPO_pdf.txt')
pdf_MAU = np.loadtxt('bigboy_pdf/MAU_pdf.txt')
pdf_WEST = np.loadtxt('bigboy_pdf/WEST_pdf.txt')
pdf_FLE = np.loadtxt('bigboy_pdf/FLE_pdf.txt')
pdf_TAS = np.loadtxt('bigboy_pdf/TAS_pdf.txt')
pdf_LWR = np.loadtxt('bigboy_pdf/LWR_pdf.txt')
pdf_CAP = np.loadtxt('bigboy_pdf/CAP_pdf.txt')
pdf_CAM = np.loadtxt('bigboy_pdf/CAM_pdf.txt')
pdf_KAI = np.loadtxt('bigboy_pdf/KAI_pdf.txt')
pdf_GOB = np.loadtxt('bigboy_pdf/GOB_pdf.txt')
pdf_TIM = np.loadtxt('bigboy_pdf/TIM_pdf.txt')
pdf_HSB = np.loadtxt('bigboy_pdf/HSB_pdf.txt')
pdf_BGB = np.loadtxt('bigboy_pdf/BGB_pdf.txt')
pdf_FIO = np.loadtxt('bigboy_pdf/FIO_pdf.txt')
