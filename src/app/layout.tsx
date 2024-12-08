export default function RootLayout({ children }: { children: React.ReactNode }) {
    return (
        <html lang="en">
        <body style={{ margin: 0, padding: 0, fontFamily: "Arial, sans-serif", background: "#f0f2f5" }}>
        <div style={{ padding: "20px" }}>
            <header style={{ marginBottom: "30px", textAlign: "center" }}>
                <h1 style={{ margin: 0 }}>Stock Predictor</h1>
            </header>
            {children}
            <footer style={{ marginTop: "40px", textAlign: "center", fontSize: "0.9em", color: "#555" }}>
                <p>Powered by Next.js and Python</p>
            </footer>
        </div>
        </body>
        </html>
    );
}
