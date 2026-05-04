/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: "class",
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        olumlu: "#22c55e",
        olumsuz: "#ef4444",
        notr: "#f59e0b",
      },
    },
  },
  plugins: [],
};
