#!/usr/bin/env python
import os
import sys, traceback
import socket
import time
import random
import struct
import numpy as np
import cv
import cv2

from subprocess import call

# Our IP address
UDP_IP_HOST = "192.168.1.1"
UDP_PORT_HOST = 5123 # arbitrary

# Camera IP address and UDP port
UDP_IP_TARGET = "192.168.1.2"
UDP_PORT_TARGET = 5000

# network socket timeout in seconds
SOCKET_TIMEOUT = 2

# Location of image recordings
SAVE_DIR = "/mnt/imagestore/"

# Nb of UDP packets to receive to get one full image.
# Packets are typically 904 bytes, 80 packets is about 70kB, which is enough
# for the camera setup I used (640x480 medium quality jpeg)
NB_FRAGMENTS_TO_ACCUMULATE = 80

# filtering threshold for the construction of the delta image
ABSDIFF_THRESHOLD_LEVEL = 50

# Above this min number of pixels in the delta image, a detection is notified.
NB_DIFFERENCES_THRESHOLD_MIN = 250
# Above this max number of pixels in the delta image, this is likely a false positive,
# too much has changed (e.g. happens during the IR to normal light transition frames)
NB_DIFFERENCES_THRESHOLD_MAX = 50000

# Nb of consecutive images that will be systematically recorded once a motion detection is triggered
NB_IMAGES_CAPTURED_UPON_DETECTION = 10

# Do not beep due to motion detection more than once every X seconds
BEEP_COOLDOWN_TIME = 60.0

# Period (in seconds) used for saving a frame unconditionally for monitoring/debug purpose
MONITORING_PERIOD = 900

# Name of the directory to save image to : also impact how often a new directory is created.
CAPTURE_NAME = "%Y-%m-%d"
#CAPTURE_NAME = "%Y-%m-%d_%Hh"

def byteToInt(byteVal):
	return struct.unpack('B', byteVal[0])[0]

def sendControlPacket(packet):
	print("[CONTROL] sending %d bytes " % len(packet))
	#print("[%s] sending UDP data (%d bytes)" % (time.time(),len(packet)))
	#hex_string = "".join("Sending: ").join(" %02x" % packet[b] for b in range(0, 13))
	#print(hex_string)
	sock.sendto(packet, (UDP_IP_TARGET, UDP_PORT_TARGET))	

def sendContinuePacket(packet):
	sock.sendto(packet, (UDP_IP_TARGET, UDP_PORT_TARGET))
	
def receiveControlPacket(output):
	sock.settimeout(SOCKET_TIMEOUT)
	try:
		nbbytes, addr = sock.recvfrom_into(output, 1024)
		print("[CONTROL] received %d bytes " % nbbytes)
		#print("[%s] received UDP data (from %s)  => %d bytes " % (time.time(), addr, nbbytes))
		# if (nbbytes < 64):
			# hex_string = "".join(" %02x" % buffer[b] for b in range(0, nbbytes))
			# print(hex_string)
		return nbbytes
	except socket.timeout:
		raise socket.timeout
	except KeyboardInterrupt:
		raise
	
def dumpDebugImage(path, name, image, extension):
	params = list()
	params.append(cv.CV_IMWRITE_JPEG_QUALITY)
	params.append(50)
	image = cv2.flip(image, -1) # Flip image around both axis
	# putText args : image, text, origin, font, font scale, color, thickness, line type
	cv2.putText(image,time.strftime("%Y-%m-%d  %H:%M:%S"),(10,460), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0,0,255), 2, 8)
	cv2.imwrite(path+"/"+ time.strftime("%Hh%Mm%Ss_")+ name + extension,image, params)

def recordImage(video_file, image):
	image = cv2.flip(image, -1) # Flip image around both axis
	# putText args : image, text, origin, font, font scale, color, thickness, line type
	cv2.putText(image,time.strftime("%Y-%m-%d  %H:%M:%S"),(10,460), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0,0,255), 2, 8)
	video_file.write(image)
	
class RestartException(Exception):
	def __init__(self, msg, delay=1):
		self.msg = msg
		self.delay = delay
	def __str__(self):
		return repr(self.msg)
	
MESSAGE_43 = bytearray([0x00,0x00,0xb0,0x02,0x82,0x00,0x00,0x27,0x00,0x01,0x00,0x00,0x00,0x4d,0x61,0x63,0x49,0x50,0x3d,0x42,0x43,0x2d,0x41,0x45,0x2d,0x43,0x35,0x2d,0x37,0x43,0x2d,0x37,0x37,0x2d,0x37,0x42,0x2b,0x31,0x36,0x34,0x36,0x37,0x3b])

MESSAGE_13_1 = bytearray([0x00,0x00,0xd0,0x00,0x82,0x00,0x06,0x09,0x00,0x01,0x00,0x00,0x00])
MESSAGE_13_2 = bytearray([0x00,0x00,0xd0,0x00,0xa2,0x00,0x06,0x09,0x00,0x01,0x00,0x00,0x00])
MESSAGE_13_3 = bytearray([0x00,0x00,0xd0,0x00,0x62,0x00,0x06,0x09,0x00,0x01,0x00,0x00,0x00])

MESSAGE_212 = bytearray([0x01,0x00,0x40,0x0d,0x32,0x00,0x00,0xd0,0x00,0x51,0x01,0x00,0x00,0x69,0x64,0xd4,0xd8,0xd8,0xd2,0x8f,0x9d,0xa7,0xd9,0xd4,0x9f,0x80,0x8d,0x8c,0x86,0xc7,0x9f,0x8b,0xbf,0x80,0x8d,0x8c,0x86,0xc7,0xa4,0xb9,0xac,0xae,0xdd,0xd2,0x8f,0x9d,0xa7,0xd8,0xd4,0x87,0x8c,0x9d,0xc7,0xd9,0xd2,0x8f,0x9d,0xa7,0xdb,0xd4,0xa1,0xa2,0xb9,0xaa,0xb9,0x9b,0x8c,0x9a,0x8c,0x87,0x9d,0xc7,0xa1,0xa2,0xb9,0xaa,0xb9,0x9b,0x8c,0x9a,0x8c,0x87,0x9d,0xd2,0x86,0x99,0xa7,0xdb,0xd4,0xdc,0x98,0x8d,0xdf,0xa6,0xa3,0xdf,0xda,0xda,0xdf,0xde,0x8f,0x8f,0x8f,0xd2,0xaa,0x88,0x85,0x85,0x80,0x8d,0xd4,0xdd,0x85,0x90,0xd9,0x81,0x8f,0xdc,0x82,0xd8,0xde,0xa8,0xd9,0xd9,0xae,0xb3,0xd8,0xd0,0x8f,0xda,0x85,0xdc,0xde,0xad,0x8a,0xdf,0xda,0xda,0xdf,0xd9,0x8f,0x8f,0xd9,0xd2,0x9a,0x80,0x8d,0xa7,0xd4,0xdc,0x98,0x8d,0xdf,0xa6,0xa3,0xdf,0xda,0xda,0xdf,0xde, 0x8f,0x8f,0x8f,0xd2,0xa8,0x9a,0xaa,0x86,0x8d,0x8c,0xd4,0xda,0xda,0xde,0xd2,0xa4,0x88,0x80,0x87,0xaa,0x84,0x8d,0xd4,0xa1,0xa2,0xb6,0xbb,0xac,0xba,0xb6,0xbb,0xac,0xb8,0xd2,0x9c,0x9a,0x8c,0x9b,0xd4,0xd8,0xd0,0xdb,0xc7,0xd8,0xdf,0xd1,0xc7,0xd9,0xc7,0xda,0xda,0xd2])

MESSAGE_34 = bytearray([0x00,0x00,0x20,0x02,0x12,0x00,0x00,0x1e,0x00,0x01,0x00,0x00,0x00,0xa0,0xaa,0xa4,0xad,0xd4,0xd8,0xd2,0xba,0xac,0xb8,0xd4,0xd8,0xd2,0xbd,0xa0,0xa4,0xac,0xd4,0xd9,0xd2,0xe9])

MESSAGE_119 = bytearray([0x02,0x00,0x70,0x07,0x32,0x00,0x00,0x73,0x00,0x64,0x00,0x00,0x00,0x4d,0x61,0x80,0x87,0xaa,0x84,0x8d,0xd4,0xba,0x8c,0x9a,0x9a,0x80,0x86,0x87,0xba,0x9d,0x88,0x9b,0x9d,0xd2,0x9a,0x80,0x8d,0xa7,0xd4,0xdc,0x98,0x8d,0xdf,0xa6,0xa3,0xdf,0xda,0xda,0xdf,0xde,0x8f,0x8f,0x8f,0xd2,0x8f,0x9d,0xa7,0xd9,0xd4,0xa1,0xa2,0xb9,0xaa,0xb9,0x9b,0x8c,0x9a,0x8c,0x87,0x9d,0xc7,0xa1,0xa2,0xb9,0xaa,0xb9,0x9b,0x8c,0x9a,0x8c,0x87,0x9d,0xd2,0xaf,0xad,0xd9,0xd4,0xdb,0xdc,0xd8,0xdd,0xd0,0xdb,0xd1,0xd1,0xd2,0x8f,0x9d,0xa7,0xd8,0xd4,0x87,0x8c,0x9d,0xc7,0xd8,0xd9,0xdb,0xdc,0xd2,0xaf,0xad,0xd8,0xd4,0xd8,0xd9,0xdb,0xdc,0xd2])

# The continue packet is composed of a first part where the 0xff below get dynamically replaced by the appropriate value at runtime, and a second part that is invariable
MESSAGE_CONTINUE_BEGIN = bytearray([0x00,0x00,0xff,0x02,0x12,0x00,0x00,0xff,0x00,0x01,0x00,0x00,0x00,0xa0,0xaa,0xa4,0xad,0xd4,0xd8,0xd2,0xba,0xac,0xb8,0xd4])
MESSAGE_CONTINUE_END = bytearray([0xd2,0xbd,0xa0,0xa4,0xac,0xd4,0xd9,0xd2,0xe9])

# in the continue packet, each digit goes through this sequence
CONTINUE_LIST_2 = bytearray([0xd9,0xd8,0xdb,0xda,0xdd,0xdc,0xdf,0xde,0xd1,0xd0])

# in the continue packet, the last digit toggles between two values (e.g. 0xd8 and 0xdf)
# periodically, change the toggle set (e.g move to 0xdb/0xde)
CONTINUE_LIST_1 = bytearray([0xd8,0xdf, 0xdb,0xde, 0xda,0xd1, 0xdd,0xd0, 0xdc,0xd9, 0xdf, 0xd8, 0xde,0xdb, 0xd1,0xda, 0xd0, 0xdd, 0xd9,0xdc])

# buffer for control/initialization packets reception
buffer = bytearray(1024)

global_loop_iteration = 0

capturedImageIndex = 0
# capturePath = SAVE_DIR

sock = None
# video_out = None
captureInProgress = False

msg = bytearray()

capturePath = SAVE_DIR + time.strftime(CAPTURE_NAME)
if not os.path.exists(capturePath): 
	os.makedirs(capturePath)
video_out = cv2.VideoWriter(capturePath+"/"+'capture.avi',cv.CV_FOURCC('X','V','I','D'), 4.0, (640,480))
if not video_out.isOpened():
	print("[ERROR] unable to open video file in %s" % capturePath)
	
try:
	while True:
		try:
			global_loop_iteration +=1
			
			print("*****************************************************************")
			print("Global loop iteration #%d started on %s" % (global_loop_iteration, time.strftime("%Y-%m-%d @ %H:%M:%S")))
			print("*****************************************************************")
			
			#########################
			# VARIOUS INITIALIZATIONS
			#########################
			
			# the 7th byte in the 13 byte msg seems to be arbitrary: pick any random value for which bit 4 is not already set
			val = random.randint(0,16)
			MESSAGE_13_1[6] = val
			MESSAGE_13_2[6] = val
			MESSAGE_13_3[6] = val
			
			msg = b''
			inImage = False
			imageIndex = 0
			lastFragmentId = 0
			fragmentIndex = 0

			nbDigits = 1
			continue_index = [0,0,0,0,0]
			base_index = 0
			fragments_received = 0

			bytes=''
			latestBeepTime = 0 # To ensure that the first detection will beep
			latestImageMonitoringTime = time.time()
			socket_error = False
			nbDifferences = 0
			imagesToCapture = 0

			RGBImageCurrent = None
			RGBImagePrev = None

			GRAYImageCurrent = None
			GRAYImagePrev = None
			
			#RunningAverageImage = None

			DetectionMask = cv2.imread("detectionmask.png", cv2.CV_LOAD_IMAGE_GRAYSCALE)
			
			# Initialize video file output
			# capturePath = SAVE_DIR + time.strftime("%Y-%m-%d_%H")
			# if not os.path.exists(capturePath): 
				# os.makedirs(capturePath)
			# video_out = cv2.VideoWriter(capturePath+"/"+'capture.avi',cv.CV_FOURCC('X','V','I','D'), 4.0, (640,480))
			
			#######################
			# NETWORK RELATED SETUP
			#######################
			# In case this is not the first run
			if sock: 
					sock.close()
					
			# Open UDP socket to talk to camera 
			try:
				sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
				sock.bind((UDP_IP_HOST, UDP_PORT_HOST))
			except socket.error, (value,message): 
				if sock: 
					sock.close() 
				raise RestartException("Could not open socket: " + message)
			except KeyboardInterrupt:
				raise
				
		    ################################
			# CAMERA INITIALIZATION SEQUENCE
			################################
			sendControlPacket(MESSAGE_43)

			sendControlPacket(MESSAGE_13_1)
				
			for i in range(5):
				try:
					nbReceived = receiveControlPacket(buffer)
					if ((nbReceived == 13) and (buffer[4] == (MESSAGE_13_1[4] | 0b00010000)) and (buffer[6] != (MESSAGE_13_1[6] | 0b01000000))):
						print("[CONTROL] status ok")
						break
					else:
						print("[CONTROL] status ko, repeating")
						#print("status ko: repeat (buff4=%x, mess4and=%x)" % (buffer[4],(MESSAGE_13_1[4] | 0b00010000)))
						if (i==9):
							raise RestartException("Max number of status check loops reached")
				except socket.timeout:
					raise RestartException("Socket timeout 1", 15)
				except KeyboardInterrupt:
					raise

			sendControlPacket(MESSAGE_13_2)
			sendControlPacket(MESSAGE_13_3)
			try:
				nbReceived = receiveControlPacket(buffer)
				if (nbReceived != 13):
					raise RestartException("Expected 13 bytes, received %d" % nbReceived)
			except socket.timeout:
				raise RestartException("Socket timeout 2")
			except KeyboardInterrupt:
				raise

			sendControlPacket(MESSAGE_212)
			try:
				nbReceived = receiveControlPacket(buffer)
				# Sometimes the 42 bytes packet comes before the 115: discard it and re-read
				if(nbReceived == 42):
					print("[CONTROL] received 42 early, re-reading")
					nbReceived = receiveControlPacket(buffer)
				if (nbReceived != 155):
					raise RestartException("Expected 155 bytes, received %d" % nbReceived)
			except socket.timeout:
				raise RestartException("Socket timeout 3")
			except KeyboardInterrupt:
				raise
				
			sendControlPacket(MESSAGE_34)

			sendControlPacket(MESSAGE_119)
			try:
				nbReceived = receiveControlPacket(buffer)
				# Sometimes the 42 bytes packet comes at this point: discard it and re-read
				if(nbReceived == 42):
					print("[CONTROL] received 42 late, re-reading")
					nbReceived = receiveControlPacket(buffer)
				if (nbReceived != 368):
					raise RestartException("Expected 368 bytes, received %d" % nbReceived)
			except socket.timeout:
				raise RestartException("Socket timeout 4")
			except KeyboardInterrupt:
				raise
				
			# reception packet TAPA 410 & paquet 42 bytes
			#receiveControlPacket(buffer)

		    ############################
			# BEGIN IMAGE RECEPTION LOOP
			############################

			while socket_error == False:
				sock.settimeout(SOCKET_TIMEOUT)

				# Receive UDP fragment
				try:	
					chunk = sock.recv(1024)
				except KeyboardInterrupt:
					raise
				except: 
					socket_error = True
					print("[DATA] Sock.recv error")
					continue
					
				nbbytes = len(chunk)
				fragments_received += 1
				fragmentIndex += 1

				if (fragments_received <= NB_FRAGMENTS_TO_ACCUMULATE):
					# Filter out any potential non-image-data packets (e.g. 13 bytes statuses)
					if nbbytes >= 17:
						# First frame / Start of Image : get rid of the 15 bytes header
						if (chunk[15] == '\xff') and (chunk[16] == '\xd8'):
							lastFragmentId = byteToInt(chunk[0])
							msg+= chunk[15:]
						# additional data fragment : just drop the 4 bytes header and concatenate to already received data
						else:
							# Check for sequence number continuity
							if ((byteToInt(chunk[0]) == lastFragmentId+1) or (byteToInt(chunk[0]) == 0) and (lastFragmentId==255)):
								msg += chunk[4:]
							# If we lost a fragment, no point in continuing accumulating data for this frame so restart another data grab
							else:
								msg = b''
								fragments_received = 0	
							# Keep track of sequence number
							lastFragmentId = byteToInt(chunk[0])
					# If we received an unexpected packet in the middle of the image data, something is wrong : just drop the ongoing image capture & restart
					else:
						msg = b''
						fragments_received = 0	
				else:
					# We now normally have enough data so that a full image is present in the buffer: search for SOI and EOI markers 
					# SOI = 0xffd8
					# EOI = 0xffd9
					SOI_index = -1
					EOI_index = -1
					for index in range(0,len(msg)-1):
						if (msg[index] == '\xff'):
							if msg[index+1] == '\xd8':
								SOI_index = index
								for index in range(index+2,len(msg)-1):
									if (msg[index] == '\xff'):
										if msg[index+1] == '\xd9':
											EOI_index = index
											break
								break

					if SOI_index!=-1 and EOI_index!=-1:
						# A complete image was indeed found in the data buffer : isolate the image data in a dedicated buffer
						# Keep the rest of data for next iterations
						jpeg = msg[SOI_index:EOI_index+2]
						msg = msg[EOI_index+2:]

						try:
							now = time.time()
							
							# Convert raw data buffer to OpenCV image format
							RGBImageNext = cv2.imdecode(np.fromstring(jpeg, dtype=np.uint8),cv2.CV_LOAD_IMAGE_COLOR)
							
							# Create a grayscale version of the image for analysis
							GRAYImageNext = cv2.cvtColor(RGBImageNext,cv2.COLOR_BGR2GRAY)
							
							# Launch motion detection 
							if (imageIndex > 2):
								d1 = cv2.absdiff(GRAYImagePrev, GRAYImageNext)
								d2 = cv2.absdiff(GRAYImageCurrent, GRAYImageNext)
								#result = np.zeros(d1.shape)
								result = cv2.bitwise_and(d1, d2)
								#masked_result = np.zeros(result.shape)
								masked_result = cv2.bitwise_and(result, DetectionMask)
								thresholded = cv2.threshold(masked_result, ABSDIFF_THRESHOLD_LEVEL, 255, cv2.THRESH_BINARY)
								nbDifferences = cv2.countNonZero(thresholded[1])
								
								# DEBUG / Experimental
								#cv2.accumulateWeighted(GRAYImageNext,RunningAverageImage,0.5)
								#avg_image = cv2.convertScaleAbs(RunningAverageImage)
							
							# Skip detection for the first two frames
							else:
								#RunningAverageImage = np.float32(GRAYImageNext)
								nbDifferences = 0

							motion_detected = False
							
							if ( (nbDifferences > NB_DIFFERENCES_THRESHOLD_MIN) and (nbDifferences < NB_DIFFERENCES_THRESHOLD_MAX) and (imageIndex != 0)):
								
								# Play a sound to signal the detection, but not too often.
								if (now - latestBeepTime) > BEEP_COOLDOWN_TIME:
									call(["aplay", "-q", "beepbeep.wav"])
									print("[CONTROL] Playing detection sound on %s" % time.strftime("%Y-%m-%d @ %H:%M:%S"))
									latestBeepTime = now
									
								motion_detected = True
								
								# Handle the case of a new/fresh detection
								if captureInProgress == False:
									
									captureInProgress = True
									print("[DATA] image %d on %s: MOTION DETECTED" % (imageIndex, time.strftime("%Y-%m-%d @ %H:%M:%S")))
									
									# Reset sequence-internal image index
									capturedImageIndex = 0
														
									# Create new dir for storing this sequence
									capturePath = SAVE_DIR + time.strftime(CAPTURE_NAME)
									# This is a new day : open new video file (create directory if required)
									if not os.path.isfile(capturePath+"/"+'capture.avi'): 
										if not os.path.exists(capturePath):
											print("[CONTROL] Creating directory %s" % capturePath)
											os.makedirs(capturePath)
									
										# reinitialize daily video file (note : will overwrite exiting one from the same day if it already exists, because
										# there is no way to reopen an existing file with VideoWriter)
										video_out = cv2.VideoWriter(capturePath+"/"+'capture.avi',cv.CV_FOURCC('X','V','I','D'), 30.0, (640,480))
										if not video_out.isOpened():
											print("[ERROR] unable to open video file %s" % capturePath+"/"+'capture.avi')
										else:
											print("[CONTROL] opened new video capture file %s" % capturePath+"/"+'capture.avi')
									else:
										print("[CONTROL] Reusing existing video file %s" % capturePath+"/"+'capture.avi')
									
									# DEBUG
									dumpDebugImage(capturePath, "absdif_thresholded", thresholded[1], ".png")
									#dumpDebugImage(capturePath, "masked_result", masked_result, ".png")
									#dumpDebugImage(capturePath, "averageimg", avg_image, ".png")									
									#dumpDebugImage(capturePath, "capture"+str(capturedImageIndex), RGBImagePrev, ".jpg")
									#capturedImageIndex +=1
									#dumpDebugImage(capturePath, "capture"+str(capturedImageIndex), RGBImageCurrent, ".jpg")
									#capturedImageIndex +=1
										
									# Record the last 2 images (N-2 and N-1) and the difference image that triggered the detection
									recordImage(video_out, RGBImagePrev)
									recordImage(video_out, RGBImageCurrent)
										
								# In any case, start (or re-start) the recording of N consecutive images
								imagesToCapture = NB_IMAGES_CAPTURED_UPON_DETECTION
							
							if (motion_detected or imagesToCapture > 0):							
								# DEBUG
								# dumpDebugImage(capturePath, "capture"+str(capturedImageIndex), RGBImageNext, ".jpg")
								
								recordImage(video_out, RGBImageNext)
								imagesToCapture -=1
								
								if imagesToCapture == 0:
									captureInProgress = False
										
								capturedImageIndex +=1
							
							# Unconditionally save a frame periodically
							if (now - latestImageMonitoringTime) > MONITORING_PERIOD:
								path = SAVE_DIR +time.strftime("%Y-%m-%d")+ "/daily_monitoring"
								if not os.path.exists(path): 
									print("[CONTROL] daily monitoring: creating folder %s" % path)
									os.makedirs(path)
								dumpDebugImage(path, "monitoring", RGBImageNext, ".jpg")
								# Update monitoring capture time
								latestImageMonitoringTime = now
							
							# Update images for next analysis iteration
							RGBImagePrev = RGBImageCurrent
							RGBImageCurrent = RGBImageNext
											
							GRAYImagePrev = GRAYImageCurrent
							GRAYImageCurrent = GRAYImageNext
						
						except KeyboardInterrupt:
							raise
						except:
							exc_type, exc_value, exc_traceback = sys.exc_info()
							traceback.print_exception(exc_type, exc_value, exc_traceback,limit=2, file=sys.stdout)
							del exc_traceback
							pass
						
						# Log a trace every 5 min or so (5*60s*3img/sec)
						if (imageIndex % 900 == 0):
							 print("[DATA] Still alive %s, image index %d" % (time.strftime("%Y-%m-%d @ %H:%M:%S"), imageIndex))

						imageIndex += 1	
			#		else:
			#			print("no image found in stream among %d bytes"% len(msg))
					
					# Restart another data grab
					msg = b''
					fragments_received = 0

				####################################
				# MANAGE "CONTINUE" PACKETS SEQUENCE
				####################################
				
				# Send out a feedback message every 5 fragments received, to tell the camera to keep sending frames.
				if (fragmentIndex%5) == 0:
					tmp = bytearray()
						
					if (nbDigits == 1):
						MESSAGE_CONTINUE_BEGIN[2] = 0x20
						MESSAGE_CONTINUE_BEGIN[7] = 0x1e
						tmp.append(CONTINUE_LIST_1[base_index+continue_index[0]])
						continue_index[0] += 1
				
						if continue_index[0] == 2:
							nbDigits += 1
							continue_index[1] = 1 # start at d8
							continue_index[0] = 0
							
					elif (nbDigits == 2):
						MESSAGE_CONTINUE_BEGIN[2] = 0x30
						MESSAGE_CONTINUE_BEGIN[7] = 0x1f
						tmp.append(CONTINUE_LIST_2[continue_index[1]])
						tmp.append(CONTINUE_LIST_1[base_index+continue_index[0]])
						
						continue_index[0] += 1
									
						if continue_index[0] == 2:
							continue_index[1] += 1
							continue_index[0] = 0
						
						if continue_index[1] == len(CONTINUE_LIST_2):
							nbDigits += 1
							continue_index[2] = 1 # start at d8
							continue_index[1] = 0 # start at d9
							continue_index[0] = 0
							
					elif (nbDigits == 3):
						MESSAGE_CONTINUE_BEGIN[2] = 0x40
						MESSAGE_CONTINUE_BEGIN[7] = 0x20
						tmp.append(CONTINUE_LIST_2[continue_index[2]])
						tmp.append(CONTINUE_LIST_2[continue_index[1]])
						tmp.append(CONTINUE_LIST_1[base_index+continue_index[0]])
						
						# update digit 0
						continue_index[0] += 1
						
						# update digit 1
						if continue_index[0] == 2:
							continue_index[1] += 1
							continue_index[0] = 0
						# update digit 2
						if continue_index[1] == len(CONTINUE_LIST_2):
							continue_index[2] += 1
							continue_index[1] = 0 # start at d9
							continue_index[0] = 0			
						# check for adding one more digit
						if continue_index[2] == len(CONTINUE_LIST_2):
							nbDigits += 1
							continue_index[3] = 1 # start at d8
							continue_index[2] = 0 # start at d9
							continue_index[1] = 0 # start at d9
							continue_index[0] = 0 # start at d9
							
					elif (nbDigits == 4):
						MESSAGE_CONTINUE_BEGIN[2] = 0x50
						MESSAGE_CONTINUE_BEGIN[7] = 0x21
						tmp.append(CONTINUE_LIST_2[continue_index[3]])
						tmp.append(CONTINUE_LIST_2[continue_index[2]])
						tmp.append(CONTINUE_LIST_2[continue_index[1]])
						tmp.append(CONTINUE_LIST_1[base_index+continue_index[0]])
						
						# update digit 0
						continue_index[0] += 1
						
						# update digit 1
						if continue_index[0] == 2:
							continue_index[1] += 1
							continue_index[0] = 0
							
						# update digit 2
						if continue_index[1] == len(CONTINUE_LIST_2):
							continue_index[2] += 1
							continue_index[1] = 0 # start at d9
							continue_index[0] = 0	
							
						# update digit 3
						if continue_index[2] == len(CONTINUE_LIST_2):
							continue_index[3] += 1 # start at d8
							continue_index[2] = 0 # start at d9
							continue_index[1] = 0 # start at d9
							continue_index[0] = 0 # start at d9
							
						# check for adding one more digit
						if continue_index[3] == len(CONTINUE_LIST_2):
							nbDigits += 1
							continue_index[4] = 1 # start at d8
							continue_index[3] = 0 # start at d9
							continue_index[2] = 0 # start at d9
							continue_index[1] = 0 # start at d9	
							
					elif (nbDigits == 5):
						MESSAGE_CONTINUE_BEGIN[2] = 0x60
						MESSAGE_CONTINUE_BEGIN[7] = 0x22
						tmp.append(CONTINUE_LIST_2[continue_index[4]])
						tmp.append(CONTINUE_LIST_2[continue_index[3]])
						tmp.append(CONTINUE_LIST_2[continue_index[2]])
						tmp.append(CONTINUE_LIST_2[continue_index[1]])
						tmp.append(CONTINUE_LIST_1[base_index+continue_index[0]])
						
						# update digit 0
						continue_index[0] += 1
						
						# update digit 1
						if continue_index[0] == 2:
							continue_index[1] += 1
							continue_index[0] = 0
							
						# update digit 2
						if continue_index[1] == len(CONTINUE_LIST_2):
							continue_index[2] += 1
							continue_index[1] = 0 # start at d9
							continue_index[0] = 0	
							
						# update digit 3
						if continue_index[2] == len(CONTINUE_LIST_2):
							continue_index[3] += 1 # start at d8
							continue_index[2] = 0 # start at d9
							continue_index[1] = 0 # start at d9
							continue_index[0] = 0 # start at d9			
						
						# update digit 4
						if continue_index[3] == len(CONTINUE_LIST_2):
							continue_index[4] += 1 # start at d8
							continue_index[3] = 0 # start at d9
							continue_index[2] = 0 # start at d9
							continue_index[1] = 0 # start at d9
							continue_index[0] = 0 # start at d9	
						
						if continue_index[4] == len(CONTINUE_LIST_2):
							# restart sequence
							#print("RESTARTING SEQUENCE")
							nbDigits = 1	
					# horrible reverse-engineered condition to restart sequence at 1 digit
					if len(tmp) == 5 and tmp.startswith(b'\xdf\xdc\xdd\xd0'):
						nbDigits = 1
					# horrible experimentally-determined condition to change the toggle data set for the last byte	
					if (fragmentIndex % 100 == 0):
						base_index = (base_index + 2) % 20  
						
					packet = MESSAGE_CONTINUE_BEGIN + tmp + MESSAGE_CONTINUE_END
					sendContinuePacket(packet)
					
		except RestartException as resExc:
			print("[ERROR] restarting global loop due to exception: %s" % resExc.msg)
			# Let the camera breathe a bit before trying again
			time.sleep(resExc.delay)
			pass
		except KeyboardInterrupt:
			raise
		#end of global loop
	# end of global try	

	
except KeyboardInterrupt:
	print("[CONTROL] manually interrupted, %s" % time.strftime("%Y-%m-%d @ %H:%M:%S")) 
except NameError as n:
	print("[ERROR] NameError %s" % n)
except:
	exc_type, exc_value, exc_traceback = sys.exc_info()
	traceback.print_exception(exc_type, exc_value, exc_traceback,limit=2, file=sys.stdout)	
	del exc_traceback
	
print("[CONTROL] exiting surveillance")
