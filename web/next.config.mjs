/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  webpack: (config) => {
    config.resolve.alias = {
      ...config.resolve.alias,
      "@tensorflow/tfjs": "@tensorflow/tfjs",
    };
    return config;
  },
};

export default nextConfig;
