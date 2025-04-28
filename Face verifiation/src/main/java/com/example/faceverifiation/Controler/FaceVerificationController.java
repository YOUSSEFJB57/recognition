package com.example.faceverifiation.Controler;

import com.example.faceverifiation. service.FaceVerificationService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

@RestController
@RequestMapping("/api")
public class FaceVerificationController {

    private final FaceVerificationService faceVerificationService;

    public FaceVerificationController(FaceVerificationService faceVerificationService) {
        this.faceVerificationService = faceVerificationService;
    }

    @PostMapping("/detect")
    public ResponseEntity<String> detectFace(@RequestParam MultipartFile img) {
        try {
            boolean isFace = faceVerificationService.isFace(img.getBytes());
            return ResponseEntity.ok("Face detected: " + isFace);
        } catch (Exception e) {
            return ResponseEntity.internalServerError().body("Error: " + e.getMessage());
        }
    }
}
