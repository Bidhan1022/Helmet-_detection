import React, { useState, useRef } from "react";

const API_URL = "http://127.0.0.1:8000";

interface Counts {
  helmet: number;
  no_helmet: number;
  number_plate: number;
  rider: number;
}

interface ImageResult {
  success: boolean;
  image_url: string;
  image_base64?: string;
  counts: Counts;
}

interface VideoResult {
  success: boolean;
  video_url: string;
  output_filename: string;
  counts: Counts;
}

export default function App() {

  // =========================
  // MODE
  // =========================

  const [mode, setMode] = useState<"image" | "video">("image");


  // =========================
  // IMAGE
  // =========================

  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imageResult, setImageResult] = useState<ImageResult | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const imageInputRef = useRef<HTMLInputElement>(null);


  // =========================
  // VIDEO
  // =========================

  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [videoResult, setVideoResult] = useState<VideoResult | null>(null);
  const videoInputRef = useRef<HTMLInputElement>(null);


  // =========================
  // SETTINGS
  // =========================

  const [imageConfidence, setImageConfidence] = useState(0.25);
  const [imageIou, setImageIou] = useState(0.45);

  const [videoConfidence, setVideoConfidence] = useState(0.25);
  const [videoIou, setVideoIou] = useState(0.45);

  const [advancedOpen, setAdvancedOpen] = useState(false);


  // =========================
  // PROCESSING
  // =========================

  const [processing, setProcessing] = useState(false);
  const [error, setError] = useState("");


  // =========================
  // IMAGE SELECT
  // =========================

  const handleImageSelect = (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {

    const file = event.target.files?.[0];

    if (!file) return;

    const allowed = [
      "image/jpeg",
      "image/jpg",
      "image/png"
    ];

    if (!allowed.includes(file.type)) {
      setError("Please select JPG, JPEG or PNG image.");
      return;
    }

    setError("");

    setImageFile(file);
    setImageResult(null);

    const previewUrl = URL.createObjectURL(file);
    setImagePreview(previewUrl);
  };


  // =========================
  // VIDEO SELECT
  // =========================

  const handleVideoSelect = (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {

    const file = event.target.files?.[0];

    if (!file) return;

    const extension =
      "." + file.name.split(".").pop()?.toLowerCase();

    const allowed = [
      ".mp4",
      ".avi",
      ".mov"
    ];

    if (!allowed.includes(extension)) {
      setError("Please select MP4, AVI or MOV video.");
      return;
    }

    setError("");

    setVideoFile(file);
    setVideoResult(null);
  };


  // =========================
  // IMAGE DETECTION
  // =========================

  const detectImage = async () => {

    if (!imageFile) return;

    setProcessing(true);
    setError("");

    const formData = new FormData();
    formData.append("file", imageFile);

    try {

      const response = await fetch(
        `${API_URL}/predict/image?conf=${imageConfidence}&iou=${imageIou}`,
        {
          method: "POST",
          body: formData
        }
      );

      if (!response.ok) {
        throw new Error("Image detection failed.");
      }

      const data: ImageResult =
        await response.json();

      setImageResult(data);

    } catch (err) {

      console.error(err);

      setError(
        "Detection failed. Make sure FastAPI backend is running."
      );

    } finally {

      setProcessing(false);

    }
  };


  // =========================
  // VIDEO DETECTION
  // =========================

  const detectVideo = async () => {

    if (!videoFile) return;

    setProcessing(true);
    setError("");

    const formData = new FormData();
    formData.append("file", videoFile);

    try {

      const response = await fetch(
        `${API_URL}/predict/video?conf=${videoConfidence}&iou=${videoIou}`,
        {
          method: "POST",
          body: formData
        }
      );

      if (!response.ok) {
        throw new Error("Video detection failed.");
      }

      const data: VideoResult =
        await response.json();

      setVideoResult(data);

    } catch (err) {

      console.error(err);

      setError(
        "Video detection failed. Make sure FastAPI backend is running."
      );

    } finally {

      setProcessing(false);

    }
  };


  // =========================
  // IMAGE RESULT URL
  // =========================

  const getImageResultUrl = () => {

    if (!imageResult) return "";

    if (imageResult.image_base64) {
      return imageResult.image_base64;
    }

    return `${API_URL}${imageResult.image_url}`;
  };


  // =========================
  // VIDEO RESULT URL
  // =========================

  const getVideoResultUrl = () => {

    if (!videoResult) return "";

    return `${API_URL}${videoResult.video_url}`;
  };


  // =========================
  // RENDER
  // =========================

  return (

    <div className="min-h-screen bg-[#090d16] text-white">

      {/* =========================
          HEADER
      ========================= */}

      <header className="border-b border-slate-800 bg-[#0f172a]">

        <div className="max-w-5xl mx-auto px-5 py-5">

          <h1 className="text-2xl font-bold">
            Helmet Detection
          </h1>

          <p className="text-sm text-slate-400 mt-1">
            AI-based Helmet, Rider and Number Plate Detection
          </p>

        </div>

      </header>


      {/* =========================
          MAIN
      ========================= */}

      <main className="max-w-5xl mx-auto px-5 py-8">


        {/* MODE BUTTONS */}

        <div className="flex gap-3 mb-6">

          <button
            onClick={() => setMode("image")}
            className={`px-5 py-2.5 rounded-lg font-semibold ${
              mode === "image"
                ? "bg-blue-600 text-white"
                : "bg-slate-800 text-slate-300"
            }`}
          >
            Image
          </button>

          <button
            onClick={() => setMode("video")}
            className={`px-5 py-2.5 rounded-lg font-semibold ${
              mode === "video"
                ? "bg-blue-600 text-white"
                : "bg-slate-800 text-slate-300"
            }`}
          >
            Video
          </button>

        </div>


        {/* ERROR */}

        {error && (

          <div className="mb-5 p-4 rounded-lg bg-red-500/10 border border-red-500/30 text-red-300">
            {error}
          </div>

        )}


        {/* =========================
            IMAGE MODE
        ========================= */}

        {mode === "image" && (

          <div className="space-y-5">


            {/* UPLOAD */}

            <div className="bg-[#111827] border border-slate-800 rounded-xl p-6">

              <h2 className="text-lg font-bold mb-4">
                Upload Image
              </h2>

              <input
                ref={imageInputRef}
                type="file"
                accept=".jpg,.jpeg,.png"
                onChange={handleImageSelect}
                className="hidden"
              />

              <button
                onClick={() =>
                  imageInputRef.current?.click()
                }
                className="w-full border-2 border-dashed border-slate-700 hover:border-blue-500 rounded-xl p-10 text-center text-slate-300"
              >

                {imageFile
                  ? imageFile.name
                  : "Click to select image"}

              </button>

            </div>


            {/* ADVANCED SETTINGS */}

            <div className="bg-[#111827] border border-slate-800 rounded-xl">

              <button
                onClick={() =>
                  setAdvancedOpen(!advancedOpen)
                }
                className="w-full px-5 py-4 text-left font-semibold"
              >
                ⚙ Advanced Settings
              </button>


              {advancedOpen && (

                <div className="px-5 pb-5 space-y-5">

                  {/* Confidence */}

                  <div>

                    <div className="flex justify-between text-sm mb-2">

                      <span>
                        Confidence
                      </span>

                      <span className="text-blue-400">
                        {imageConfidence.toFixed(2)}
                      </span>

                    </div>

                    <input
                      type="range"
                      min="0.05"
                      max="0.95"
                      step="0.05"
                      value={imageConfidence}
                      onChange={(e) =>
                        setImageConfidence(
                          parseFloat(e.target.value)
                        )
                      }
                      className="w-full"
                    />

                  </div>


                  {/* IoU */}

                  <div>

                    <div className="flex justify-between text-sm mb-2">

                      <span>
                        IoU
                      </span>

                      <span className="text-blue-400">
                        {imageIou.toFixed(2)}
                      </span>

                    </div>

                    <input
                      type="range"
                      min="0.10"
                      max="0.90"
                      step="0.05"
                      value={imageIou}
                      onChange={(e) =>
                        setImageIou(
                          parseFloat(e.target.value)
                        )
                      }
                      className="w-full"
                    />

                  </div>

                </div>

              )}

            </div>


            {/* DETECT */}

            <button
              onClick={detectImage}
              disabled={!imageFile || processing}
              className="w-full py-3 rounded-lg bg-blue-600 hover:bg-blue-500 disabled:opacity-50 font-semibold"
            >

              {processing
                ? "Detecting..."
                : "Detect"}

            </button>


            {/* RESULT */}

            {imageResult && (

              <div className="bg-[#111827] border border-slate-800 rounded-xl p-5">

                <h2 className="text-lg font-bold mb-4">
                  Result
                </h2>


                {/* COUNTS */}

                <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-5">

                  <CountCard
                    title="Helmet"
                    value={imageResult.counts.helmet}
                    color="text-green-400"
                  />

                  <CountCard
                    title="No Helmet"
                    value={imageResult.counts.no_helmet}
                    color="text-red-400"
                  />

                  <CountCard
                    title="Rider"
                    value={imageResult.counts.rider}
                    color="text-blue-400"
                  />

                  <CountCard
                    title="Number Plate"
                    value={imageResult.counts.number_plate}
                    color="text-yellow-400"
                  />

                </div>


                {/* IMAGE */}

                <img
                  src={getImageResultUrl()}
                  alt="Helmet Detection Result"
                  className="w-full max-h-[600px] object-contain rounded-lg bg-black"
                />


                {/* DOWNLOAD */}

                <a
                  href={getImageResultUrl()}
                  download="helmet_detection_result.jpg"
                  className="block text-center mt-4 py-3 rounded-lg bg-slate-700 hover:bg-slate-600 font-semibold"
                >
                  Download Result
                </a>

              </div>

            )}

          </div>

        )}


        {/* =========================
            VIDEO MODE
        ========================= */}

        {mode === "video" && (

          <div className="space-y-5">


            {/* UPLOAD */}

            <div className="bg-[#111827] border border-slate-800 rounded-xl p-6">

              <h2 className="text-lg font-bold mb-4">
                Upload Video
              </h2>

              <input
                ref={videoInputRef}
                type="file"
                accept=".mp4,.avi,.mov"
                onChange={handleVideoSelect}
                className="hidden"
              />

              <button
                onClick={() =>
                  videoInputRef.current?.click()
                }
                className="w-full border-2 border-dashed border-slate-700 hover:border-blue-500 rounded-xl p-10 text-center text-slate-300"
              >

                {videoFile
                  ? videoFile.name
                  : "Click to select video"}

              </button>

            </div>


            {/* ADVANCED SETTINGS */}

            <div className="bg-[#111827] border border-slate-800 rounded-xl">

              <button
                onClick={() =>
                  setAdvancedOpen(!advancedOpen)
                }
                className="w-full px-5 py-4 text-left font-semibold"
              >
                ⚙ Advanced Settings
              </button>


              {advancedOpen && (

                <div className="px-5 pb-5 space-y-5">

                  <div>

                    <div className="flex justify-between text-sm mb-2">

                      <span>
                        Confidence
                      </span>

                      <span className="text-blue-400">
                        {videoConfidence.toFixed(2)}
                      </span>

                    </div>

                    <input
                      type="range"
                      min="0.05"
                      max="0.95"
                      step="0.05"
                      value={videoConfidence}
                      onChange={(e) =>
                        setVideoConfidence(
                          parseFloat(e.target.value)
                        )
                      }
                      className="w-full"
                    />

                  </div>


                  <div>

                    <div className="flex justify-between text-sm mb-2">

                      <span>
                        IoU
                      </span>

                      <span className="text-blue-400">
                        {videoIou.toFixed(2)}
                      </span>

                    </div>

                    <input
                      type="range"
                      min="0.10"
                      max="0.90"
                      step="0.05"
                      value={videoIou}
                      onChange={(e) =>
                        setVideoIou(
                          parseFloat(e.target.value)
                        )
                      }
                      className="w-full"
                    />

                  </div>

                </div>

              )}

            </div>


            {/* DETECT */}

            <button
              onClick={detectVideo}
              disabled={!videoFile || processing}
              className="w-full py-3 rounded-lg bg-blue-600 hover:bg-blue-500 disabled:opacity-50 font-semibold"
            >

              {processing
                ? "Processing..."
                : "Detect"}

            </button>


            {/* RESULT */}

            {videoResult && (

              <div className="bg-[#111827] border border-slate-800 rounded-xl p-5">

                <h2 className="text-lg font-bold mb-4">
                  Result
                </h2>


                {/* COUNTS */}

                <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-5">

                  <CountCard
                    title="Helmet"
                    value={videoResult.counts.helmet}
                    color="text-green-400"
                  />

                  <CountCard
                    title="No Helmet"
                    value={videoResult.counts.no_helmet}
                    color="text-red-400"
                  />

                  <CountCard
                    title="Rider"
                    value={videoResult.counts.rider}
                    color="text-blue-400"
                  />

                  <CountCard
                    title="Number Plate"
                    value={videoResult.counts.number_plate}
                    color="text-yellow-400"
                  />

                </div>


                {/* VIDEO */}

                <video
                  src={getVideoResultUrl()}
                  controls
                  className="w-full max-h-[600px] rounded-lg bg-black"
                />


                {/* DOWNLOAD */}

                <a
                  href={getVideoResultUrl()}
                  download={videoResult.output_filename}
                  className="block text-center mt-4 py-3 rounded-lg bg-slate-700 hover:bg-slate-600 font-semibold"
                >
                  Download Result
                </a>

              </div>

            )}

          </div>

        )}

      </main>

    </div>
  );
}


// =========================
// COUNT CARD
// =========================

function CountCard({
  title,
  value,
  color
}: {
  title: string;
  value: number;
  color: string;
}) {

  return (

    <div className="bg-slate-900 border border-slate-800 rounded-lg p-4 text-center">

      <p className="text-sm text-slate-400">
        {title}
      </p>

      <p className={`text-3xl font-bold mt-1 ${color}`}>
        {value}
      </p>

    </div>

  );
}