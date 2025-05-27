import type { NextConfig } from "next";

const nextConfig: NextConfig = {
	webpack(config) {
		config.module.rules.push({
			test: /\.(glsl|vs|fs|vert|frag)$/,
			use: [
				{
					loader: 'raw-loader',
					options: { esModule: false }
				},
			],
		});
		return config;
	},
} as NextConfig;

export default nextConfig;
