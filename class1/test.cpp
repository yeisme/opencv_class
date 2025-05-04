
#include <opencv2/opencv.hpp>
#include "MvCameraControl.h"

using namespace cv;

enum CONVERT_TYPE
{
	OpenCV_Mat = 0,    // ch:Matͼ���ʽ | en:Mat format
	OpenCV_IplImage = 1,    // ch:IplImageͼ���ʽ | en:IplImage format
};

// ch:��ʾö�ٵ����豸��Ϣ | en:Print the discovered devices' information
void PrintDeviceInfo(MV_CC_DEVICE_INFO* pstMVDevInfo)
{
	if (NULL == pstMVDevInfo)
	{
		printf("    NULL info.\n\n");
		return;
	}

	// ��ȡͼ������֡��֧��GigE��U3V�豸
	if (MV_GIGE_DEVICE == pstMVDevInfo->nTLayerType)
	{
		int nIp1 = ((pstMVDevInfo->SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24);
		int nIp2 = ((pstMVDevInfo->SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16);
		int nIp3 = ((pstMVDevInfo->SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8);
		int nIp4 = (pstMVDevInfo->SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff);

		// ch:��ʾIP���豸�� | en:Print current ip and user defined name
		printf("    IP: %d.%d.%d.%d\n", nIp1, nIp2, nIp3, nIp4);
		printf("    UserDefinedName: %s\n", pstMVDevInfo->SpecialInfo.stGigEInfo.chUserDefinedName);
		printf("    Device Model Name: %s\n\n", pstMVDevInfo->SpecialInfo.stGigEInfo.chModelName);
	}
	else if (MV_USB_DEVICE == pstMVDevInfo->nTLayerType)
	{
		printf("    UserDefinedName: %s\n", pstMVDevInfo->SpecialInfo.stUsb3VInfo.chUserDefinedName);
		printf("    Device Model Name: %s\n\n", pstMVDevInfo->SpecialInfo.stUsb3VInfo.chModelName);
	}
	else
	{
		printf("    Not support.\n\n");
	}
}

// ch:����������RGBתΪBGR | en:Convert pixel arrangement from RGB to BGR
void RGB2BGR(unsigned char* pRgbData, unsigned int nWidth, unsigned int nHeight)
{
	if (NULL == pRgbData)
	{
		return;
	}

	// red��blue���ݻ���
	for (unsigned int j = 0; j < nHeight; j++)
	{
		for (unsigned int i = 0; i < nWidth; i++)
		{
			unsigned char red = pRgbData[j * (nWidth * 3) + i * 3];
			pRgbData[j * (nWidth * 3) + i * 3] = pRgbData[j * (nWidth * 3) + i * 3 + 2];
			pRgbData[j * (nWidth * 3) + i * 3 + 2] = red;
		}
	}
}

// ch:֡����ת��ΪMat��ʽͼƬ������ | en:Convert data stream to Mat format then save image
bool Convert2Mat(MV_FRAME_OUT_INFO_EX* pstImageInfo, unsigned char* pData)
{
	if (NULL == pstImageInfo || NULL == pData)
	{
		printf("NULL info or data.\n");
		return false;
	}

	cv::Mat srcImage;

	if (PixelType_Gvsp_Mono8 == pstImageInfo->enPixelType)                // Mono8����
	{
		srcImage = cv::Mat(pstImageInfo->nHeight, pstImageInfo->nWidth, CV_8UC1, pData);
	}
	else if (PixelType_Gvsp_RGB8_Packed == pstImageInfo->enPixelType)     // RGB8����
	{
		// Mat�������и�ʽΪBGR����Ҫת��
		RGB2BGR(pData, pstImageInfo->nWidth, pstImageInfo->nHeight);
		srcImage = cv::Mat(pstImageInfo->nHeight, pstImageInfo->nWidth, CV_8UC3, pData);
	}
	else
	{
		/* Bayer ��ʽת��mat��ʽ�ķ���:
		1. ʹ������������ǰ ���� MV_CC_ConvertPixelType ��PixelType_Gvsp_BayerRG8 ��Bayer��ʽת���� PixelType_Gvsp_BGR8_Packed
		2. �ο����� ��BGRת��Ϊ mat��ʽ
		*/

		printf("Unsupported pixel format\n");
		return false;
	}

	if (NULL == srcImage.data)
	{
		printf("Creat Mat failed.\n");
		return false;
	}

	try
	{
		// ch:����MatͼƬ | en:Save converted image in a local file
		cv::imwrite("Image_Mat.bmp", srcImage);
	}
	catch (cv::Exception& ex)
	{
		fprintf(stderr, "Exception in saving mat image: %s\n", ex.what());
	}

	Mat mRisizeImg;
	resize(srcImage, mRisizeImg, Size(srcImage.cols / 5, srcImage.rows / 5));

	imshow("Image", mRisizeImg);
	waitKey(10);

	srcImage.release();

	return true;
}

// ch:֡����ת��ΪIplImage��ʽͼƬ������ | en:Convert data stream in Ipl format then save image

bool GetDeviceInfo(MV_CC_DEVICE_INFO_LIST& stDeviceList)
{
	int nRet = MV_OK;


	memset(&stDeviceList, 0, sizeof(MV_CC_DEVICE_INFO_LIST));

	// ch:�豸ö�� | en:Enum device
	nRet = MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, &stDeviceList);
	if (MV_OK != nRet)
	{
		printf("Enum Devices fail! nRet [0x%x]\n", nRet);
		return false;
	}

	// ch:��ʾ�豸��Ϣ | en:Show devices
	if (stDeviceList.nDeviceNum > 0)
	{
		for (unsigned int i = 0; i < stDeviceList.nDeviceNum; i++)
		{
			printf("[device %d]:\n", i);
			MV_CC_DEVICE_INFO* pDeviceInfo = stDeviceList.pDeviceInfo[i];
			if (NULL == pDeviceInfo)
			{
				return false;
			}
			PrintDeviceInfo(pDeviceInfo);
		}
	}
	else
	{
		printf("Find No Devices!\n");
		return false;
	}

	return true;
}

bool SetCamera(MV_CC_DEVICE_INFO_LIST& stDeviceList, unsigned int& nIndex)
{
	while (1)
	{
		printf("Please Input camera index(0-%d): ", stDeviceList.nDeviceNum - 1);

		if (1 == scanf_s("%d", &nIndex))
		{
			while (getchar() != '\n')
			{
				;
			}

			// �Ϸ�����
			if (nIndex >= 0 && nIndex < stDeviceList.nDeviceNum)
			{
				// �豸�������ӣ���������
				if (false == MV_CC_IsDeviceAccessible(stDeviceList.pDeviceInfo[nIndex], MV_ACCESS_Exclusive))
				{
					printf("Can't connect! ");
					continue;
				}

				break;
			}
		}
		else
		{
			while (getchar() != '\n')
			{
				;
			}
		}
	}


	return true;
}

bool InitDevice()
{

	return true;
}

int main()
{
	int nRet = MV_OK;
	void* handle = NULL;
	unsigned char* pData = NULL;
	MV_CC_DEVICE_INFO_LIST stDeviceList;
	GetDeviceInfo(stDeviceList);
	unsigned int nIndex = 0;
	unsigned int nPayloadSize = 0;
	MV_FRAME_OUT_INFO_EX stImageInfo = { 0 };

	SetCamera(stDeviceList, nIndex);


	do
	{
		// ch:�����豸��� | en:Create handle
		nRet = MV_CC_CreateHandle(&handle, stDeviceList.pDeviceInfo[nIndex]);
		if (MV_OK != nRet)
		{
			printf("Create Handle fail! nRet [0x%x]\n", nRet);
			break;
		}

		// ch:���豸 | en:Open device
		nRet = MV_CC_OpenDevice(handle);
		if (MV_OK != nRet)
		{
			printf("Open Device fail! nRet [0x%x]\n", nRet);
			break;
		}

		// ch:̽�����Packet��С��ֻ֧��GigE����� | en:Detection network optimal package size(It only works for the GigE camera)
		if (MV_GIGE_DEVICE == stDeviceList.pDeviceInfo[nIndex]->nTLayerType)
		{
			int nPacketSize = MV_CC_GetOptimalPacketSize(handle);
			if (nPacketSize > 0)
			{
				// ����Packet��С
				nRet = MV_CC_SetIntValue(handle, "GevSCPSPacketSize", nPacketSize);
				if (MV_OK != nRet)
				{
					printf("Warning: Set Packet Size fail! nRet [0x%x]!", nRet);
				}
			}
			else
			{
				printf("Warning: Get Packet Size fail! nRet [0x%x]!", nPacketSize);
			}
		}

		// ch:�رմ���ģʽ | en:Set trigger mode as off
		nRet = MV_CC_SetEnumValue(handle, "TriggerMode", 0);
		if (MV_OK != nRet)
		{
			printf("Set Trigger Mode fail! nRet [0x%x]\n", nRet);
			break;
		}

		// ch:��ȡͼ���С | en:Get payload size
		MVCC_INTVALUE stParam;
		memset(&stParam, 0, sizeof(MVCC_INTVALUE));
		nRet = MV_CC_GetIntValue(handle, "PayloadSize", &stParam);
		if (MV_OK != nRet)
		{
			printf("Get PayloadSize fail! nRet [0x%x]\n", nRet);
			break;
		}
		nPayloadSize = stParam.nCurValue;

		// ch:��ʼ��ͼ����Ϣ | en:Init image info
		memset(&stImageInfo, 0, sizeof(MV_FRAME_OUT_INFO_EX));
		pData = (unsigned char*)malloc(sizeof(unsigned char) * (nPayloadSize));
		if (NULL == pData)
		{
			printf("Allocate memory failed.\n");
			break;
		}
		memset(pData, 0, sizeof(pData));
	} while (0);

	do
	{
		// ch:��ʼȡ�� | en:Start grab image
		nRet = MV_CC_StartGrabbing(handle);
		if (MV_OK != nRet)
		{
			printf("Start Grabbing fail! nRet [0x%x]\n", nRet);
			break;
		}

		// ch:��ȡһ֡ͼ�񣬳�ʱʱ��1000ms | en:Get one frame from camera with timeout=1000ms
		nRet = MV_CC_GetOneFrameTimeout(handle, pData, nPayloadSize, &stImageInfo, 1000);
		if (MV_OK == nRet)
		{
			printf("Get One Frame: Width[%d], Height[%d], FrameNum[%d]\n",
				stImageInfo.nWidth, stImageInfo.nHeight, stImageInfo.nFrameNum);
		}
		else
		{
			printf("Get Frame fail! nRet [0x%x]\n", nRet);
			break;
		}

		// ch:ֹͣȡ�� | en:Stop grab image
		nRet = MV_CC_StopGrabbing(handle);
		if (MV_OK != nRet)
		{
			printf("Stop Grabbing fail! nRet [0x%x]\n", nRet);
			break;
		}

		int nFormat = 0;


		// ch:����ת�� | en:Convert image data
		bool bConvertRet = Convert2Mat(&stImageInfo, pData);


		// ch:��ʾת����� | en:Print result
		if (bConvertRet)
		{
			printf("OpenCV format convert finished.\n");
		}
		else
		{
			printf("OpenCV format convert failed.\n");
		}
	} while (1);

	// ch:���پ�� | en:Destroy handle
	if (handle)
	{
		// ch:�ر��豸 | en:Close device
		nRet = MV_CC_CloseDevice(handle);
		if (MV_OK != nRet)
		{
			printf("ClosDevice fail! nRet [0x%x]\n", nRet);
		}

		// ch:����Ҫת���ĸ�ʽ | en:Input the format to convert
		printf("\n[0] OpenCV_Mat\n");
		printf("[1] OpenCV_IplImage\n");

		MV_CC_DestroyHandle(handle);
		handle = NULL;
	}

	// ch:�ͷ��ڴ� | en:Free memery
	if (pData)
	{
		free(pData);
		pData = NULL;
	}

	system("pause");
	return 0;
}

