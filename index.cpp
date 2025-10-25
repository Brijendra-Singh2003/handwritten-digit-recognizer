#include "./src/predict.cpp"
#include <raylib.h>

const int HEIGHT = 720;
const int WIDTH = 1280;
const int BLOCK_SIZE = 32;
const string WEIGHTS_FILE_PATH = "weights2.txt";

int main() {
    InitWindow(WIDTH, HEIGHT, "Number Prediction");
    SetTargetFPS(60);

    mnist_model model(WEIGHTS_FILE_PATH);
    int prediction = 8;

    Rectangle input_screen {24, 24, 672, 672};
    Rectangle predict_button {720, 632, 256, 64};
    Rectangle reset_button {1000, 632, 256, 64};
    Rectangle mouse{};

    MNISTImage image;
    image.cols = 28;
    image.rows = 28;
    image.pixels = vector<uint8_t>((image.cols+1) * (image.rows+1));

    while (!WindowShouldClose()) {
        mouse.x = GetMouseX();
        mouse.y = GetMouseY();

        Color predict_button_bgcolor = CheckCollisionRecs(predict_button, mouse) ? GRAY : WHITE;
        Color reset_button_bgcolor = CheckCollisionRecs(reset_button, mouse) ? GRAY : WHITE;
        
        if(IsMouseButtonDown(MouseButton::MOUSE_BUTTON_LEFT)) {
            if(CheckCollisionRecs(input_screen, mouse)) {
                int j_mouse = (mouse.x - 24) / 24;
                int i_mouse = (mouse.y - 24) / 24;
    
                for(int i=max(i_mouse-1, 0); i<=min(i_mouse+1, 28); i++) {
                    for(int j=max(j_mouse-1, 0); j<=min(j_mouse+1, 28); j++) {
                        uint8_t &current_pixal = image.pixels[i * image.cols + j];
                        current_pixal = current_pixal >= 200 ? 255 : current_pixal + 127;
                    }
                }
            }
        }

        if(IsMouseButtonPressed(MouseButton::MOUSE_BUTTON_LEFT)) {
            if(CheckCollisionRecs(mouse, reset_button)) {
                fill(image.pixels.begin(), image.pixels.end(), 0);
            }

            if(CheckCollisionRecs(mouse, predict_button)) {
                prediction = model.predict(image);
            }
        }

        BeginDrawing();
            ClearBackground(BLACK);
            
            for(int i=0; i<image.rows; i++) {
                for(int j=0; j<image.cols; j++) {
                    uint8_t brightness = image.pixels[i * image.cols + j];
                    DrawRectangle(24 + 24*j, 24 + 24*i, 24, 24, {brightness, brightness, brightness, 255});
                }
            }
            
            DrawRectangleLines(24, 24, 24*28, 24*28, WHITE);

            DrawText(to_string(prediction).c_str(), predict_button.x + 108, 48, 632-24, WHITE);

            DrawRectangleRec(predict_button, predict_button_bgcolor);
            DrawText("Predict", predict_button.x + 48, predict_button.y + 12, 48, BLACK);

            DrawRectangleRec(reset_button, reset_button_bgcolor);
            DrawText("Reset", reset_button.x + 64, reset_button.y + 12, 48, BLACK);
        EndDrawing();
    }
    
    CloseWindow();
}