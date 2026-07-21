/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // The backend lives on Render; the frontend stays static so Vercel Hobby
  // function usage stays near zero. All calls go browser -> Render directly.
  env: {
    NEXT_PUBLIC_API_BASE: process.env.NEXT_PUBLIC_API_BASE ?? "",
  },
};

export default nextConfig;
