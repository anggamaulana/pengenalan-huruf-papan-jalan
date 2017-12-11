import cv2
import numpy as np

from matplotlib import pyplot as plt

from PIL import Image
import pytesseract


label=np.array(["J","L","S","U","N","D","A","H","T","O","B","R","M","E","P","K","G"])
training=[]
labeltraining=[]

def ContrastStretching(img_gray):
    r1=120.0
    s1=50.0
    r2=150.0
    s2=250.0
    # ==================
    r3=255.0
    s3=255.0

    # Untuk r <= 0 < r1, maka s = r . (s1 / r1)
    # Untuk r1 <= r < r2, maka s = s1 + ( (r-r1) . ((s2-s1) / (r2-r1)) )
    # Untuk r2 <= r <=(L-1), maka s = s2 + ( (r-r2) . ((L-1)-s2) / ((L-1) - r2) )

    img_gray2=np.where((img_gray<r1),255,0)

    # img_gray2=np.where((img_gray<r1) , np.floor(img_gray*(s1/r1)),
    #     np.where(((img_gray>r1)&(img_gray<r2)) , np.floor(s1+(img_gray-r1)*((s2-s1)/(r2-r1))),
    #         np.where(((img_gray>r2)&(img_gray<r3)) , np.floor(s2+(img_gray-r2)*((s3-s2)/(r3-r2))),img_gray)))
    return img_gray2.astype(np.uint8)

def StrokeSelection(orientation_matrix,roi_index):
	# h = orientation_matrix.shape[0]
	# w = orientation_matrix.shape[1]

    
	# xls=""
	# for x in range(0,w):
	# 	for h in range(0,h):
	# 		xls+=str(orientation_matrix[x,y])+","
	# 	xls+="\n"

	# file = open("roi"+str(roi_index)+".csv","w")
	# file.write(xls)
	# print orientation_matrix.shape

	# np.savetxt("roi"+str(roi_index)+".csv", orientation_matrix, delimiter=",")
	print "printed"

def classify(roi_humoment):
	# X = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]
	# Y = [ 0,   1,   1,    0,   1,   2,   2,   0,   1]

	# Z = [x for _,x in sorted(zip(Y,X))]
	# print(Z)  # ["a", "d", "h", "b", "c", "e", "i", "f", "g"]
	euclideans=[]
	for i in training:
		euclideans.append(np.sum(np.absolute(i-roi_humoment)))

	tmptraininglebel = labeltraining
	Z = [x for _,x in sorted(zip(euclideans,tmptraininglebel))]
	K=3



	# knn=Z[0:K]
	# print "knn"
	# print knn
	# bincnt = np.bincount(knn)
	# return np.argmax(bincnt)
	return Z[0]

def strokeWidth(roi):
	h=roi.shape[0]
	w=roi.shape[1]

	datawidth=np.array([])
	# print "max width"+str((0.25*float(w)))

	for y in range(0,h):
		for x in range(0,w):
			if roi[y,x]==0:
				# Horizontal width===============
				next_x=x
				previous_x=x
				width_horizontal=0
				while roi[y,next_x]==0:
					width_horizontal=width_horizontal+1
					next_x=next_x+1
					if next_x>(w-1):
						break
				while roi[y,previous_x]==0:
					width_horizontal=width_horizontal+1
					previous_x=previous_x-1
					if previous_x<=0:
						break
				# Vertical width=================
				next_y=y
				previous_y=y
				width_vertical=0
				while roi[next_y,x]==0:
					width_vertical=width_vertical+1
					next_y=next_y+1
					if next_y>(h-1):
						break
				while roi[previous_y,x]==0:
					width_vertical=width_vertical+1
					previous_y=previous_y-1
					if previous_y<=0:
						break



				datawidth=np.append(datawidth,min(width_horizontal,width_vertical,(0.5*float(w))))
	# print str(np.average(datawidth))+" / "+str(w)
	# print datawidth
	return np.var(datawidth),(np.average(datawidth)/float(w)),np.average(datawidth)

def setDataTraining(i,j,humoments):
	# print "roi "+str(i)+" "+str(j)+":" 
	if (i==0 and j==13) or (i==1 and j==29) or (i==2 and j==12):
		training.append(humoments)
		labeltraining.append("J")
	if (i==0 and j==12) or (i==1 and j==28) or (i==2 and j==11):
		training.append(humoments)
		labeltraining.append("L")
	if (i==0 and j==11) or (i==1 and j==23) or (i==5 and j==2):
		training.append(humoments)
		labeltraining.append("S")
	if (i==0 and j==10) or (i==1 and j==22) or (i==4 and j==3):
		training.append(humoments)
		labeltraining.append("U")
	if (i==0 and j==9) or (i==1 and j==25) or (i==1 and j==16):
		training.append(humoments)
		labeltraining.append("N")
	if (i==0 and j==8) or (i==2 and j==3) or (i==4 and j==9):
		training.append(humoments)
		labeltraining.append("D")
	if (i==0 and j==7) or (i==1 and j==27) or (i==1 and j==24):
		training.append(humoments)
		labeltraining.append("A")
	if (i==1 and j==26) or (i==2 and j==10) or (i==2 and j==7):
		training.append(humoments)
		labeltraining.append("H")
	if (i==1 and j==21) or (i==4 and j==5):
		training.append(humoments)
		labeltraining.append("T")
	if (i==1 and j==17) or (i==3 and j==15) or (i==3 and j==13):
		training.append(humoments)
		labeltraining.append("O")
	if (i==2 and j==8) or (i==3 and j==14) or (i==5 and j==7):
		training.append(humoments)
		labeltraining.append("B")
	if (i==2 and j==13) or (i==3 and j==7) or (i==4 and j==0):
		training.append(humoments)
		labeltraining.append("R")
	if (i==2 and j==5) or (i==3 and j==18):
		training.append(humoments)
		labeltraining.append("M")
	if (i==2 and j==0):
		training.append(humoments)
		labeltraining.append("E")
	if (i==4 and j==7):
		training.append(humoments)
		labeltraining.append("P")
	if (i==4 and j==2) or (i==5 and j==3):
		training.append(humoments)
		labeltraining.append("K")
	if (i==5 and j==10):
		training.append(humoments)
		labeltraining.append("G")



#Your image path i-e receipt path
total=0
mser = cv2.MSER_create()
for i in range(0,6):
	img = cv2.imread('datas/papan'+str(i)+'.jpg')
	vis = img.copy()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	stretch=ContrastStretching(gray)
	roi_stretch=stretch.copy()
	# print img
	ret, src2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	# print "thresh="+str(ret)
	total=total+ret

	

	mser.setMaxArea(5000)
	regions, _ = mser.detectRegions(stretch)
	# hulls = [np.int0(cv2.boxPoints(cv2.minAreaRect(p.reshape(-1, 1, 2)))) for p in regions]
	hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
	cv2.polylines(vis, hulls, 1, (0, 255, 0))

	# print "HULLS"
	# print hulls
	# cv2.drawContours(vis,hulls,-1,(0,255,0),3)


	# dst = cv2.Canny(stretch, 50, 200)
	
	# (_, cnts, _) = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	j=0
	if i==i:
		for p in regions:
			c=p.reshape(-1, 1, 2)
			# rotatedRect=cv2.minAreaRect(c)
			# box = cv2.boxPoints(rotatedRect)
			x,y,w,h = cv2.boundingRect(c)
			
			if (h>=10 and h<=200) and (w>=10 and h<=200) and (float(h)/float(w))>=0.7 and (float(h)/float(w))<=2.5:
				# print str(y)+" "+str(x)+" "+str(h)+" "+str(w)
				roi=roi_stretch[y:y+h,x:x+w]
				cv2.rectangle(vis,(x,y),(x+w,y+h),(0,0,255))
				sobelx = cv2.Sobel(roi,cv2.CV_64F,1,0,ksize=5)
				sobely = cv2.Sobel(roi,cv2.CV_64F,0,1,ksize=5)
				# np.savetxt("sobelx"+str(j)+".csv", sobelx, delimiter=",")
				# np.savetxt("sobely"+str(j)+".csv", sobely, delimiter=",")
				orientation=np.rad2deg(np.arctan(sobely,sobelx))
				# print "roi "+str(i)+" "+str(j)+":" 
				humoments=cv2.HuMoments(cv2.moments(roi)).flatten()
				setDataTraining(i,j,humoments)
				# StrokeSelection(orientation,j)
				# cv2.imshow('roi'+str(i)+" "+str(j), roi)
			# if j>5:
			# 	break
			j=j+1


		# cv2.imshow('block'+str(i), vis)
	# cv2.imshow('img'+str(i), src2)
		# cv2.imshow('stretch'+str(i), stretch)
	
	
print "TRAINING DONE"
# print training
# print labeltraining
# print (total/6)
# TESTING

for i in range(0,6):
	img = cv2.imread('../datas/papan'+str(i)+'.jpg')
	vis = img.copy()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	stretch=ContrastStretching(gray)
	roi_stretch=stretch.copy()
	# print img
	ret, src2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	# print "thresh="+str(ret)
	total=total+ret

	

	mser.setMaxArea(5000)
	regions, _ = mser.detectRegions(stretch)
	# hulls = [np.int0(cv2.boxPoints(cv2.minAreaRect(p.reshape(-1, 1, 2)))) for p in regions]
	# hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
	# cv2.polylines(vis, hulls, 1, (0, 255, 0))

	# print "HULLS"
	# print hulls
	# cv2.drawContours(vis,hulls,-1,(0,255,0),3)


	# dst = cv2.Canny(stretch, 50, 200)
	
	# (_, cnts, _) = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	j=0
	if i==i:
		for p in regions:
			c=p.reshape(-1, 1, 2)
			# rotatedRect=cv2.minAreaRect(c)
			# box = cv2.boxPoints(rotatedRect)
			x,y,w,h = cv2.boundingRect(c)
			
			if (h>=10 and h<=200) and (w>=10 and h<=200) and (float(h)/float(w))>=0.7 and (float(h)/float(w))<=2.5:
				# print str(y)+" "+str(x)+" "+str(h)+" "+str(w)
				roi=roi_stretch[y:y+h,x:x+w]
				varianceStroke,ratioMeanWidth,mn = strokeWidth(roi)
				strokeVariance=round(varianceStroke)
				if ratioMeanWidth<0.25 and ratioMeanWidth<0.455:
					continue

				print "variance : "+str(strokeVariance)
				print "mean : "+str(mn)
				print "width : "+str(w)
				print "meanWidthRAtio : "+str(round(ratioMeanWidth,2))
				cv2.rectangle(vis,(x,y),(x+w,y+h),(0,255,0))
				sobelx = cv2.Sobel(roi,cv2.CV_64F,1,0,ksize=5)
				sobely = cv2.Sobel(roi,cv2.CV_64F,0,1,ksize=5)
				# np.savetxt("sobelx"+str(j)+".csv", sobelx, delimiter=",")
				# np.savetxt("sobely"+str(j)+".csv", sobely, delimiter=",")
				orientation=np.rad2deg(np.arctan(sobely,sobelx))
				 
				humoments=cv2.HuMoments(cv2.moments(roi)).flatten()
				font = cv2.FONT_HERSHEY_SIMPLEX
				class_letter = classify(humoments)
				cv2.putText(vis,str(round(ratioMeanWidth,2)),(x,y+5), font, 0.3, (255,255,0), 1, cv2.LINE_AA)
				cv2.putText(vis,str(class_letter),(x,y+50), font, 0.8, (255,255,0), 2, cv2.LINE_AA)
				print "roi "+str(i)+" "+str(j)+" : classify as "+class_letter
				StrokeSelection(orientation,j)
				# cv2.imshow('roi'+str(i)+" "+str(j), roi)
			# if j>5:
			# 	break
			j=j+1
	cv2.imshow('block'+str(i), vis)
	cv2.imshow('stretch'+str(i), stretch)


print "TESTING DONE"
cv2.waitKey(0)
