"use client"

import { useState } from "react"

type VideoPlayerProps = {
  videoUrl: string
}

export default function VideoPlayer({ videoUrl }: VideoPlayerProps) {
  const [isLoading, setIsLoading] = useState(true)

  return (
    <div className="mb-8 rounded-xl overflow-hidden shadow-lg animated-border">
      <div className="aspect-w-16 aspect-h-9 bg-black relative">
        {isLoading && (
          <div className="absolute inset-0 flex items-center justify-center bg-gray-900">
            <div className="w-16 h-16 border-4 border-green-500 border-t-transparent rounded-full animate-spin"></div>
          </div>
        )}
        <iframe
          src={videoUrl}
          title="Educational Video"
          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
          allowFullScreen
          className="w-full h-full"
          onLoad={() => setIsLoading(false)}
        ></iframe>
      </div>
    </div>
  )
}

