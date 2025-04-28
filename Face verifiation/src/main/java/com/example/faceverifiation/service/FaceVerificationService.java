package com.example.faceverifiation.service;

import ai.onnxruntime.*;
import org.apache.commons.io.IOUtils;
import org.springframework.stereotype.Service;
import jakarta.annotation.PostConstruct;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.Graphics2D;
import java.awt.Image;
import java.io.InputStream;
import java.io.ByteArrayInputStream;
import java.nio.FloatBuffer;
import java.util.Collections;

@Service
public class FaceVerificationService {

    private OrtEnvironment env;
    private OrtSession session;

    @PostConstruct
    public void init() throws Exception {
        env = OrtEnvironment.getEnvironment();

        // Load model from resources
        try (InputStream is = getClass().getResourceAsStream("/models/face_classifier_full.onnx")) {
            if (is == null) {
                throw new Exception("Model file not found!");
            }
            byte[] modelBytes = IOUtils.toByteArray(is);
            session = env.createSession(modelBytes, new OrtSession.SessionOptions());
        }
    }

    public boolean isFace(byte[] imageBytes) throws Exception {
        float[] inputData = preprocessImage(imageBytes);

        // 1 × 3 × 64 × 64  tensor
        OnnxTensor inputTensor = OnnxTensor.createTensor(
                env, FloatBuffer.wrap(inputData), new long[]{1, 3, 64, 64});

        OrtSession.Result output = session.run(
                Collections.singletonMap("input", inputTensor));

        float raw = ((float[][]) output.get(0).getValue())[0][0];

        // --- New: adaptively get probability -----------------------------
        float probability;
        if (raw >= 0f && raw <= 1f) {          // already a probability
            probability = raw;
        } else {                               // logit → sigmoid
            probability = (float)(1 / (1 + Math.exp(-raw)));
        }
        // -----------------------------------------------------------------

        return probability >= 0.5f;            // final decision
    }


    // Preprocessing function
    private float[] preprocessImage(byte[] imageBytes) throws Exception {
        // Read image
        BufferedImage originalImage = ImageIO.read(new ByteArrayInputStream(imageBytes));
        if (originalImage == null) {
            throw new Exception("Invalid image format!");
        }

        // Resize to 64x64
        Image resizedImage = originalImage.getScaledInstance(64, 64, Image.SCALE_SMOOTH);
        BufferedImage outputImage = new BufferedImage(64, 64, BufferedImage.TYPE_INT_RGB);
        Graphics2D g2d = outputImage.createGraphics();
        g2d.drawImage(resizedImage, 0, 0, null);
        g2d.dispose();

        // Normalize to [-1, 1] and format into [Channels, Height, Width]
        float[] data = new float[3 * 64 * 64];

        int idx = 0;
        for (int y = 0; y < 64; y++) {
            for (int x = 0; x < 64; x++) {
                int rgb = outputImage.getRGB(x, y);

                int r = (rgb >> 16) & 0xFF;
                int g = (rgb >> 8) & 0xFF;
                int b = rgb & 0xFF;

                data[idx] = (r / 255.0f - 0.5f) * 2.0f;         // Red
                data[idx + 64 * 64] = (g / 255.0f - 0.5f) * 2.0f; // Green
                data[idx + 2 * 64 * 64] = (b / 255.0f - 0.5f) * 2.0f; // Blue

                idx++;
            }
        }

        return data;
    }
}
