#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//������  
void saltNoise(cv::Mat img, int n)
{
    int x, y;
    for (int i = 0; i < n / 2; i++)
    {
        x = std::rand() % img.cols;
        y = std::rand() % img.rows;
        //�Ҷ�ͼ
        if (img.type() == CV_8UC1)
        {
            img.at<uchar>(y, x) = 255;
        }
        //��ɫͼ��
        else if (img.type() == CV_8UC3)
        {
            img.at<cv::Vec3b>(y, x)[0] = 255;
            img.at<cv::Vec3b>(y, x)[1] = 255;
            img.at<cv::Vec3b>(y, x)[2] = 255;
        }
    }
}

//������  
void pepperNoise(cv::Mat img, int n)
{
    int x, y;
    for (int i = 0; i < n / 2; i++)
    {
        x = std::rand() % img.cols;
        y = std::rand() % img.rows;
        if (img.type() == CV_8UC1)
        {
            img.at<uchar>(y, x) = 0;
        }
        else if (img.type() == CV_8UC3)
        {
            img.at<cv::Vec3b>(y, x)[0] = 0;
            img.at<cv::Vec3b>(y, x)[1] = 0;
            img.at<cv::Vec3b>(y, x)[2] = 0;
        }
    }
}

// ��ֵ�˲���
uchar medianFilter(cv::Mat img, int row, int col, int kernelSize)
{
    std::vector<uchar> pixels;
    for (int y = -kernelSize / 2; y <= kernelSize / 2; y++)
    {
        for (int x = -kernelSize / 2; x <= kernelSize / 2; x++)
        {
            pixels.push_back(img.at<uchar>(row + y, col + x));
        }
    }
    sort(pixels.begin(), pixels.end());
    auto med = pixels[kernelSize * kernelSize / 2];
    return med;
}

// ����Ӧ��ֵ�˲���
uchar adaptiveMedianFilter(cv::Mat& img, int row, int col, int kernelSize, int maxSize)
{
    std::vector<uchar> pixels;
    for (int y = -kernelSize / 2; y <= kernelSize / 2; y++)
    {
        for (int x = -kernelSize / 2; x <= kernelSize / 2; x++)
        {
            pixels.push_back(img.at<uchar>(row + y, col + x));
        }
    }

    sort(pixels.begin(), pixels.end());

    auto min = pixels[0];
    auto max = pixels[kernelSize * kernelSize - 1];
    auto med = pixels[kernelSize * kernelSize / 2];
    auto zxy = img.at<uchar>(row, col);
    if (med > min && med < max)
    {
        // to B
        if (zxy > min && zxy < max)
            return zxy;
        else
            return med;
    }
    else
    {
        kernelSize += 2;
        if (kernelSize <= maxSize)
            return adaptiveMedianFilter(img, row, col, kernelSize, maxSize);// ���󴰿ڳߴ磬����A���̡�
        else
            return med;
    }
}

//�Ľ�������Ӧ�˲���1
uchar improvedAdaptiveMedianFilter1(cv::Mat& img, int row, int col, int kernelSize, int maxSize) {
    std::vector<uchar> pixels;
    for (int y = -kernelSize / 2; y <= kernelSize / 2; y++)
    {
        for (int x = -kernelSize / 2; x <= kernelSize / 2; x++)
        {
            pixels.push_back(img.at<uchar>(row + y, col + x));
        }
    }

    sort(pixels.begin(), pixels.end());

    auto min = pixels[0];
    auto max = pixels[kernelSize * kernelSize - 1];
    auto med = pixels[kernelSize * kernelSize / 2];
    auto zxy = img.at<uchar>(row, col);

    //δ��������Ⱦ�ĵ�
    std::vector<uchar>purePixels;
    for (auto pixel : pixels) {
        if (pixel<max && pixel>min) {
            purePixels.push_back(pixel);
        }
    }

    if (purePixels.size()>0)
    {
        //����purePixels����ֵ
        int PPMed = purePixels[purePixels.size() / 2];
        //�ж�zxy�Ƿ�Ϊ����
        if (zxy<max && zxy>min) {
            return zxy;
        }
        else {
            return PPMed;
        }
    }
    else
    {
        kernelSize += 2;
        if (kernelSize <= maxSize)
            return improvedAdaptiveMedianFilter1(img, row, col, kernelSize, maxSize);// ���󴰿ڳߴ�
        else
            return med;
    }
}

//����MSD
uchar MSD(cv::Mat& img, int row, int col, int maxSize) {
    std::vector<uchar> pixels;
    for (int y = -maxSize / 2; y <= maxSize / 2; y++)
    {
        for (int x = -maxSize / 2; x <= maxSize / 2; x++)
        {
            pixels.push_back(img.at<uchar>(row + y, col + x));
        }
    }
    sort(pixels.begin(), pixels.end());
    auto min = pixels[0];
    auto max = pixels[maxSize * maxSize - 1];
    auto med = pixels[maxSize * maxSize / 2];
    auto zxy = img.at<uchar>(row, col);

    //δ��������Ⱦ�ĵ�
    std::vector<uchar>purePixels;
    for (auto pixel : pixels) {
        if (pixel<max && pixel>min) {
            purePixels.push_back(pixel);
        }
    }
    //if (purePixels.size() == 0)return zxy;
    //���㼸�ξ���
    std::vector<uchar>MSDSet;
    for (auto purePixel : purePixels) {
        if (purePixel >= zxy) {
            MSDSet.push_back(purePixel - zxy);
        }
        else {
            MSDSet.push_back(zxy - purePixel);
        }
    }
    sort(MSDSet.begin(), MSDSet.end());
    return MSDSet[0];
}

//�Ľ�������Ӧ�˲���2
uchar improvedAdaptiveMedianFilter2(cv::Mat& img, int row, int col, int kernelSize, int maxSize,int k) {
    std::vector<uchar> pixels;
    for (int y = -kernelSize / 2; y <= kernelSize / 2; y++)
    {
        for (int x = -kernelSize / 2; x <= kernelSize / 2; x++)
        {
            pixels.push_back(img.at<uchar>(row + y, col + x));
        }
    }

    sort(pixels.begin(), pixels.end());

    auto min = pixels[0];
    auto max = pixels[kernelSize * kernelSize - 1];
    auto med = pixels[kernelSize * kernelSize / 2];
    auto zxy = img.at<uchar>(row, col);
    if (zxy > max && zxy < min) {
        return zxy;
    }
    else {
        //δ��������Ⱦ�ĵ�
        std::vector<uchar>purePixels;
        for (auto pixel : pixels) {
            if (pixel<max && pixel>min) {
                purePixels.push_back(pixel);
            }
        }

        if (purePixels.size() > 0)
        {
            //����purePixels����ֵ
            int PPMed = purePixels[purePixels.size() / 2];
            uchar MSDValue = MSD(img, row, col, maxSize);
            if (MSDValue < k)return zxy;
            else return PPMed;
        }
        else
        {
            kernelSize += 2;
            if (kernelSize <= maxSize)
                return improvedAdaptiveMedianFilter2(img, row, col, kernelSize, maxSize,k);// ���󴰿ڳߴ�
            else
                return zxy;
        }
    }
}



int main()
{
    int minSize = 3;
    int maxSize = 7;
    //�Ľ��㷨2 MSD��ֵ
    int k = 2;
    int imgSize;
    double noiseIntensity = 1;
    cv::Mat img;
    img = cv::imread("test.jpg");
    imgSize = img.cols * img.rows;
    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    cv::imshow("src", img);
    saltNoise(img, imgSize * noiseIntensity);
    pepperNoise(img, imgSize * noiseIntensity);
    cv::imshow("noise", img);
    cv::Mat temp1 = img.clone();
    cv::Mat temp2 = img.clone();
    cv::Mat temp3 = img.clone();

    // ����Ӧ��ֵ�˲�
    cv::Mat img1;
    // ��չͼ��ı߽�
    cv::copyMakeBorder(img, img1, maxSize / 2, maxSize / 2, maxSize / 2, maxSize / 2, cv::BorderTypes::BORDER_REFLECT);
    // ͼ��ѭ��
    for (int j = maxSize / 2; j < img1.rows - maxSize / 2; j++)
    {
        for (int i = maxSize / 2; i < img1.cols - maxSize / 2; i++)
        {
            img1.at<uchar>(j, i) = adaptiveMedianFilter(img1, j, i, minSize, maxSize);
        }
    }
    cv::imshow("adaptiveMedianFilter", img1);

    // ��ֵ�˲�
    cv::Mat img2;
    int kernelSize = 3;
    cv::copyMakeBorder(temp1, img2, kernelSize / 2, kernelSize / 2, kernelSize / 2, kernelSize / 2, cv::BorderTypes::BORDER_REFLECT);
    for (int j = kernelSize / 2; j < img2.rows - kernelSize / 2; j++)
    {
        for (int i = kernelSize / 2; i < img2.cols - kernelSize / 2; i++)
        {
            img2.at<uchar>(j, i) = medianFilter(img2, j, i, kernelSize);
        }
    }
    cv::imshow("medianFilter", img2);

    // �Ľ�������Ӧ��ֵ�˲�1
    cv::Mat img3;
    // ��չͼ��ı߽�
    cv::copyMakeBorder(temp2, img3, maxSize / 2, maxSize / 2, maxSize / 2, maxSize / 2, cv::BorderTypes::BORDER_REFLECT);
    // ͼ��ѭ��
    for (int j = maxSize / 2; j < img3.rows - maxSize / 2; j++)
    {
        for (int i = maxSize / 2; i < img3.cols - maxSize / 2; i++)
        {
            img3.at<uchar>(j, i) = improvedAdaptiveMedianFilter1(img3, j, i, minSize, maxSize);
        }
    }
    cv::imshow("improvedAdaptiveMedianFilter1", img3);

    // �Ľ�������Ӧ��ֵ�˲�2
    cv::Mat img4;
    // ��չͼ��ı߽�
    cv::copyMakeBorder(temp3, img4, maxSize / 2, maxSize / 2, maxSize / 2, maxSize / 2, cv::BorderTypes::BORDER_REFLECT);
    // ͼ��ѭ��
    for (int j = maxSize / 2; j < img4.rows - maxSize / 2; j++)
    {
        for (int i = maxSize / 2; i < img4.cols - maxSize / 2; i++)
        {
            img4.at<uchar>(j, i) = improvedAdaptiveMedianFilter2(img4, j, i, minSize, maxSize,k);
        }
    }
    cv::imshow("improvedAdaptiveMedianFilter2", img4);

    cv::waitKey();
    cv::destroyAllWindows();

    return 0;
}