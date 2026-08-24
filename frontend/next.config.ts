import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
  agentRules: false, // avoid a second, redundant CLAUDE.md nested inside frontend/
};

export default nextConfig;
