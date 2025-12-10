#### 🚀 Turning Images Into Music with Google Vertex AI (Gemini + Lyria)

This project converts any image into AI-generated instrumental music using Google Cloud’s Vertex AI platform.
Perfect for creators, photographers, and anyone who wants visuals to have their own soundtrack.

I can imagine a street photographer using this as a mobile app capturing images of objects while the app automatically generates background music for each shot. Later, when assembling the final album, the music is already perfectly aligned with the photos.

#### 🎯 In this video, we’ll explore

---
#### 🎶🚀 Image-to-Music with Vertex AI: How It Works

📸 → 🧠 **Gemini** analyzes the image (mood, tempo, dynamics, instruments)  
📝 → 🎵 **Lyria** converts that description into a high-quality WAV soundtrack

---

#### 🖥️🎧 Running the App: Live Demo

**See the workflow in action:**

- ⚡ Launch the app
- 📤 Upload an image and get a matching music description  
- 🎧 Play the generated WAV file or download it  
- 🔄 Generated items remain visible even after refresh


#### 👉 Links & Resources

- [Vertex AI](https://cloud.google.com/vertex-ai)
- [Lyria](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/model-reference/lyria-music-generation)

---

#### 🚀 Clone and Run

```bash
# Clone the repository
git clone https://github.com/Ashot72/Image-to-Music

# Navigate into the project directory
cd Image-to-Music

# Copy the example `.env` file and add your project ID
cp env.example .env

# Place your `service-account-key.json` file in the project root directory.

# Install dependencies
npm install

# Start the development server
npm start

# The app will be available at http://localhost:3000
```

📺 **Video:** [Watch on YouTube](https://youtu.be/dqnQK0ymzLA)
