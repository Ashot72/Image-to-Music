// Web server for Image-to-Music interface
import dotenv from 'dotenv';
import express, { Request, Response } from 'express';
import multer from 'multer';
import cors from 'cors';
import fs from 'fs';
import path from 'path';
import { VertexAI } from '@google-cloud/vertexai';
import { GoogleAuth } from 'google-auth-library';

// Load environment variables
dotenv.config();

// Type definitions
type MulterFile = Express.Multer.File;

// Configuration constants
const CONFIG = {
    PROJECT_ID: process.env.PROJECT_ID,
    LOCATION: process.env.LOCATION,
    PORT: process.env.PORT || 3000,
    MAX_FILE_SIZE: 10 * 1024 * 1024, // 10MB
} as const;

const PATHS = {
    SERVICE_ACCOUNT_KEY: path.resolve(process.cwd(), 'service-account-key.json'),
    UPLOADS: path.join(process.cwd(), 'uploads'),
    OUTPUTS: path.join(process.cwd(), 'outputs'),
    TEXTS: path.join(process.cwd(), 'texts'),
} as const;

// File type constants
const IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.gif', '.webp'] as const;
const AUDIO_EXTENSIONS = ['.wav'] as const;

const MIME_TYPES: Record<string, string> = {
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.png': 'image/png',
    '.gif': 'image/gif',
    '.webp': 'image/webp',
};

// Utility functions
function ensureDirectoryExists(dirPath: string): void {
    if (!fs.existsSync(dirPath)) {
        fs.mkdirSync(dirPath, { recursive: true });
    }
}

function setupDirectories(): void {
    ensureDirectoryExists(PATHS.UPLOADS);
    ensureDirectoryExists(PATHS.OUTPUTS);
    ensureDirectoryExists(PATHS.TEXTS);
}

function setupAuthentication(): void {
    if (!process.env.GOOGLE_APPLICATION_CREDENTIALS) {
        if (!fs.existsSync(PATHS.SERVICE_ACCOUNT_KEY)) {
            throw new Error(`Service account key file not found at: ${PATHS.SERVICE_ACCOUNT_KEY}`);
        }
        process.env.GOOGLE_APPLICATION_CREDENTIALS = PATHS.SERVICE_ACCOUNT_KEY;
    }
}

// Initialize directories and authentication
setupDirectories();
setupAuthentication();

// Initialize Vertex AI
const vertexAI = new VertexAI({
    project: CONFIG.PROJECT_ID,
    location: CONFIG.LOCATION,
});

// Express app setup
const app = express();
app.use(cors());
app.use(express.json());
app.use(express.static('public'));
app.use('/uploads', express.static(PATHS.UPLOADS));
app.use('/outputs', express.static(PATHS.OUTPUTS));
app.use('/texts', express.static(PATHS.TEXTS));

// Multer configuration
const storage = multer.diskStorage({
    destination: (
        _req: Request,
        _file: MulterFile,
        cb: (error: Error | null, destination: string) => void
    ) => {
        cb(null, PATHS.UPLOADS);
    },
    filename: (
        _req: Request,
        file: MulterFile,
        cb: (error: Error | null, filename: string) => void
    ) => {
        const originalName = path.parse(file.originalname).name;
        const ext = path.extname(file.originalname);
        cb(null, `${originalName}${ext}`);
    },
});

const fileFilter: multer.Options['fileFilter'] = (
    _req: Request,
    file: MulterFile,
    cb: multer.FileFilterCallback
): void => {
    const ext = path.extname(file.originalname).toLowerCase();
    const isValidExtension = IMAGE_EXTENSIONS.includes(ext as (typeof IMAGE_EXTENSIONS)[number]);
    const isValidMimeType = file.mimetype.startsWith('image/');

    if (isValidExtension && isValidMimeType) {
        cb(null, true);
    } else {
        cb(new Error('Only image files are allowed!'));
    }
};

const upload = multer({
    storage,
    limits: { fileSize: CONFIG.MAX_FILE_SIZE },
    fileFilter,
});

// File utility functions
function fileToGenerativePart(filePath: string, mimeType: string) {
    if (!fs.existsSync(filePath)) {
        throw new Error(`File not found at path: ${filePath}`);
    }
    const fileBuffer = fs.readFileSync(filePath);
    return {
        inlineData: {
            data: Buffer.from(fileBuffer).toString('base64'),
            mimeType,
        },
    };
}

function getMimeType(filePath: string): string {
    const ext = path.extname(filePath).toLowerCase();
    return MIME_TYPES[ext] || 'image/png';
}

// Prompt constants
const ANALYSIS_PROMPT = `Perform a comprehensive analysis of this image. Examine:

1. **Visual Elements**: What objects, people, animals, or subjects are present? What is the environment or setting?
2. **Emotional Indicators**: Analyze facial expressions, body language, postures, and gestures. What emotions are being conveyed?
3. **Visual Atmosphere**: Consider lighting (bright/dark/harsh/soft), color palette (warm/cool/vibrant/muted), and overall composition.
4. **Action & Movement**: Is the scene static or dynamic? Is there movement, action, or tension? What is the speed and intensity of any action?
5. **Context & Narrative**: What story or situation is being depicted? What is happening in the scene?
6. **Energy Level**: Assess the overall energy and intensity - is it calm, energetic, aggressive, peaceful, chaotic, or dramatic?

Based on this comprehensive analysis, generate a detailed instrumental music description that accurately matches the scene's true emotional character, energy level, and atmosphere. The music must reflect the actual intensity and mood - do not default to pleasant or neutral descriptions.

Specify each on a new line: Mood: (emotional tone), Tempo: (speed: very slow/slow/moderate/fast/very fast), Dynamics: (volume/intensity: very soft/soft/moderate/loud/very loud), and Main instruments: (list instruments). Be specific about musical characteristics that match the scene's energy and emotion.

Keep the description concise. Focus on the most important musical elements that capture the scene's essence.

Output only the music description text.`;

// AI generation functions
async function generateMusicPromptFromImage(imagePath: string): Promise<string> {
    const mimeType = getMimeType(imagePath);
    const imagePart = fileToGenerativePart(imagePath, mimeType);

    const geminiModel = vertexAI.preview.getGenerativeModel({ model: 'gemini-2.5-flash' });
    const contents = [{ role: 'user', parts: [imagePart, { text: ANALYSIS_PROMPT }] }];

    const response = await geminiModel.generateContent({ contents });

    if (!response?.response) {
        throw new Error('Invalid response from Gemini API');
    }

    const { candidates } = response.response;
    if (!candidates || candidates.length === 0) {
        throw new Error('No candidates in Gemini response');
    }

    const candidate = candidates[0];
    const parts = candidate?.content?.parts;
    if (!parts || parts.length === 0) {
        throw new Error('Invalid candidate structure');
    }

    const musicPrompt = parts[0].text?.trim();
    if (!musicPrompt) {
        throw new Error('Analysis failed: Gemini could not generate a music prompt');
    }

    return musicPrompt;
}

async function generateMusicWithLyria(prompt: string, outputPath: string): Promise<void> {
    const apiEndpoint = `https://${CONFIG.LOCATION}-aiplatform.googleapis.com/v1/projects/${CONFIG.PROJECT_ID}/locations/${CONFIG.LOCATION}/publishers/google/models/lyria-002:predict`;

    const auth = new GoogleAuth({
        keyFilename: PATHS.SERVICE_ACCOUNT_KEY,
        scopes: ['https://www.googleapis.com/auth/cloud-platform'],
    });
    const client = await auth.getClient();
    const accessToken = await client.getAccessToken();

    const response = await fetch(apiEndpoint, {
        method: 'POST',
        headers: {
            Authorization: `Bearer ${accessToken.token}`,
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            instances: [
                {
                    prompt,
                    negative_prompt: 'vocals',
                },
            ],
            parameters: {
                sample_count: 1,
            },
        }),
    });

    if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Lyria API error: ${response.status} ${response.statusText}. ${errorText}`);
    }

    const responseData = (await response.json()) as {
        predictions?: Array<{ audioContent?: string; bytesBase64Encoded?: string }>;
    };
    const predictions = responseData.predictions;

    if (!predictions || predictions.length === 0) {
        throw new Error('Lyria returned no audio prediction');
    }

    const audioContentBase64 = predictions[0].bytesBase64Encoded || predictions[0].audioContent;
    if (!audioContentBase64) {
        throw new Error('No audio data in prediction');
    }

    const audioBuffer = Buffer.from(audioContentBase64, 'base64');
    fs.writeFileSync(outputPath, audioBuffer);
}

// File name utility functions
function getBaseName(filename: string): string {
    const parsed = path.parse(filename);
    const nameWithoutExt = parsed.name;
    const lastDashIndex = nameWithoutExt.lastIndexOf('-');

    // Check if last part after dash is a timestamp (numeric)
    if (lastDashIndex > 0) {
        const suffix = nameWithoutExt.substring(lastDashIndex + 1);
        if (/^\d+$/.test(suffix)) {
            return nameWithoutExt.substring(0, lastDashIndex);
        }
    }
    return nameWithoutExt;
}

function readTextFile(filePath: string): string {
    if (!fs.existsSync(filePath)) {
        return '';
    }
    try {
        return fs.readFileSync(filePath, 'utf-8').trim();
    } catch (error) {
        console.error(`Error reading text file ${filePath}:`, error);
        return '';
    }
}

function saveTextFile(filePath: string, content: string): void {
    try {
        fs.writeFileSync(filePath, content, 'utf-8');
    } catch (error) {
        console.error(`Error saving text file ${filePath}:`, error);
    }
}

// File matching functions
function getFilesByExtension(dirPath: string, extensions: readonly string[]): string[] {
    if (!fs.existsSync(dirPath)) {
        return [];
    }
    return fs.readdirSync(dirPath).filter((file: string) => {
        const ext = path.extname(file).toLowerCase();
        return extensions.includes(ext);
    });
}

function createFileMap(
    files: string[],
    getBaseNameFn: (file: string) => string,
    dirPath: string
): Map<string, string> {
    const map = new Map<string, string>();
    files.forEach((file: string) => {
        const baseName = getBaseNameFn(file);
        if (!map.has(baseName)) {
            map.set(baseName, file);
        } else {
            // If we already have a file with this base name, keep the one with the latest modification time
            const existingFile = map.get(baseName)!;
            const existingPath = path.join(dirPath, existingFile);
            const newPath = path.join(dirPath, file);
            const existingMtime = fs.existsSync(existingPath)
                ? fs.statSync(existingPath).mtimeMs
                : 0;
            const newMtime = fs.existsSync(newPath) ? fs.statSync(newPath).mtimeMs : 0;
            if (newMtime > existingMtime) {
                map.set(baseName, file);
            }
        }
    });
    return map;
}

interface MatchedFile {
    imageUrl: string;
    audioUrl: string;
    name: string;
    prompt?: string;
}

function matchFiles(): MatchedFile[] {
    const uploadFiles = getFilesByExtension(PATHS.UPLOADS, IMAGE_EXTENSIONS);
    const outputFiles = getFilesByExtension(PATHS.OUTPUTS, AUDIO_EXTENSIONS);

    const imageMap = createFileMap(uploadFiles, getBaseName, PATHS.UPLOADS);
    const audioMap = createFileMap(outputFiles, (file) => path.parse(file).name, PATHS.OUTPUTS);

    const matchedFiles: MatchedFile[] = [];
    const allBaseNames = new Set([...imageMap.keys(), ...audioMap.keys()]);

    allBaseNames.forEach((baseName) => {
        const imageFile = imageMap.get(baseName);
        const audioFile = audioMap.get(baseName);

        if (imageFile && audioFile) {
            const textFilePath = path.join(PATHS.TEXTS, `${baseName}.txt`);
            const prompt = readTextFile(textFilePath);

            matchedFiles.push({
                imageUrl: `/uploads/${imageFile}`,
                audioUrl: `/outputs/${audioFile}`,
                name: baseName,
                prompt: prompt || undefined,
            });
        }
    });

    return matchedFiles.sort((a, b) => {
        // Use file modification time for sorting (newest first)
        const imagePathA = path.join(PATHS.UPLOADS, a.imageUrl.split('/').pop() || '');
        const imagePathB = path.join(PATHS.UPLOADS, b.imageUrl.split('/').pop() || '');

        const mtimeA = fs.existsSync(imagePathA) ? fs.statSync(imagePathA).mtimeMs : 0;
        const mtimeB = fs.existsSync(imagePathB) ? fs.statSync(imagePathB).mtimeMs : 0;

        // Sort by modification time descending (newest first)
        return mtimeB - mtimeA;
    });
}

// API Routes
app.get('/api/files', (_req: Request, res: Response) => {
    try {
        const files = matchFiles();
        res.json({ files });
    } catch (error: any) {
        console.error('Error listing files:', error);
        res.status(500).json({
            error: error.message || 'Failed to list files',
            details: error.toString(),
        });
    }
});

app.post('/api/generate', upload.single('image'), async (req: Request, res: Response) => {
    try {
        if (!req.file) {
            return res.status(400).json({ error: 'No image file provided' });
        }

        const imagePath = req.file.path;
        const imageUrl = `/uploads/${req.file.filename}`;
        const imageBaseName = path.parse(req.file.originalname).name;
        const outputFilename = `${imageBaseName}.wav`;
        const outputPath = path.join(PATHS.OUTPUTS, outputFilename);
        const outputUrl = `/outputs/${outputFilename}`;

        // Generate music prompt from image
        const musicPrompt = await generateMusicPromptFromImage(imagePath);

        // Generate music with Lyria
        await generateMusicWithLyria(musicPrompt, outputPath);

        // Save prompt text to .txt file
        const textFilePath = path.join(PATHS.TEXTS, `${imageBaseName}.txt`);
        saveTextFile(textFilePath, musicPrompt);

        res.json({
            success: true,
            imageUrl,
            prompt: musicPrompt,
            audioUrl: outputUrl,
            filename: outputFilename,
        });
    } catch (error: any) {
        console.error('Error generating music:', error);
        res.status(500).json({
            error: error.message || 'Failed to generate music',
            details: error.toString(),
        });
    }
});

// Start server
app.listen(CONFIG.PORT, () => {
    console.log(`🚀 Server running on http://localhost:${CONFIG.PORT}`);
    console.log(`📁 Uploads directory: ${PATHS.UPLOADS}`);
    console.log(`📁 Outputs directory: ${PATHS.OUTPUTS}`);
    console.log(`📁 Texts directory: ${PATHS.TEXTS}`);
});
