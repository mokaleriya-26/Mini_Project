/* === analysis.js (Wired to the Live API) === */

document.addEventListener("DOMContentLoaded", () => {

    // --- 1. GET ALL HTML ELEMENTS ---
    const niftySelect = document.getElementById("niftySelect");
    const generateBtn = document.getElementById("generateBtn");
    const placeholderCard = document.getElementById("placeholderCard");
    const fullWidthCharts = document.getElementById("fullWidthCharts");

    // News
    const newsTitle = document.getElementById("newsTitle");
    const newsList = document.getElementById("newsList");
    
    // Stats
    const statsTable = document.getElementById("statsTable");

    // Charts
    const predCtx = document.getElementById("predChart");
    const histCtx = document.getElementById("histChart");

    // Grid
    const predGrid = document.getElementById("predGrid");

    // Chart instances (to destroy them before re-drawing)
    let histChartObj;
    let predChartObj;

    // --- 2. DEFINE COMPANIES ---
    // The "value" MUST be the ticker that yfinance understands.
    // For Indian stocks, this is ".NS".
    const companies = [
        "ADANIENT.NS - Adani Enterprises",
        "ADANIPORTS.NS - Adani Ports & SEZ",
        "APOLLOHOSP.NS - Apollo Hospitals",
        "ASIANPAINT.NS - Asian Paints",
        "AXISBANK.NS - Axis Bank",
        "BAJFINANCE.NS - Bajaj Finance",
        "BHARTIARTL.NS - Bharti Airtel",
        "HDFCBANK.NS - HDFC Bank",
        "INFY.NS - Infosys",
        "ICICIBANK.NS - ICICI Bank",
        "RELIANCE.NS - Reliance Industries",
        "SBIN.NS - State Bank of India",
        "TCS.NS - Tata Consultancy Services",
        "TATAMOTORS.NS - Tata Motors",
        "TATASTEEL.NS - Tata Steel",
        "WIPRO.NS - Wipro",
        "AAPL - Apple (Test Ticker)" // For testing
    ];

    // --- 3. INITIALIZE THE PAGE ---
    
    // Populate the dropdown
    companies.forEach(c => {
        const opt = document.createElement("option");
        // Get just the ticker (e.g., "ADANIENT.NS")
        const ticker = c.split(" - ")[0]; 
        opt.value = ticker; // The value is now just the ticker
        opt.textContent = c; // The text is the full name
        niftySelect.appendChild(opt);
    });

    // Attach the click listener
    generateBtn.addEventListener("click", generateInsights);

    // --- 4. THIS IS THE NEW "HEART" OF THE SCRIPT ---
    async function generateInsights() {
        const ticker = niftySelect.value;
        if (!ticker) {
            alert("Please select a company first!");
            return;
        }

        // 1. Set loading state
        generateBtn.disabled = true;
        generateBtn.textContent = "Generating...";
        placeholderCard.style.display = "none";
        fullWidthCharts.style.display = "block"; // Show the section

        try {
            // 2. Call your real Django API
            const response = await fetch(`/api/predict/${ticker}/`);
            if (!response.ok) {
                // If API returns an error, show it
                const errorData = await response.json();
                throw new Error(errorData.error || "Failed to fetch data");
            }
            
            const data = await response.json(); // This is your REAL JSON!

            // 3. Populate all the UI elements with REAL data
            
            // Populate News
            newsTitle.textContent = `Latest News (${data.ticker})`;
            newsList.innerHTML = ""; // Clear old news
            if (data.latest_news.length > 0) {
                data.latest_news.forEach(n => {
                    const div = document.createElement("div");
                    div.className = "news-item"; // Using a class from your HTML
                    div.innerHTML = `<a href="${n.url}" target="_blank">${n.title}</a><div class="muted">${n.source}</div>`;
                    newsList.appendChild(div);
                });
            } else {
                newsList.innerHTML = `<div class="muted" style="margin-bottom:8px;">• No recent news found.</div>`;
            }

            // Populate Stats
            statsTable.innerHTML = ""; // Clear old stats
            // We use key_stats from your API
            for (const [k, v] of Object.entries(data.key_stats)) {
                let formattedValue = v;
                // Format numbers nicely
                if (typeof v === 'number') {
                    if (k === 'volume') {
                        formattedValue = v.toLocaleString(); // Add commas
                    } else {
                        formattedValue = `₹${v.toFixed(2)}`; // Add currency
                    }
                }
                const keyName = k.replace("_", " ").toUpperCase(); // "last_close" -> "LAST CLOSE"
                const row = document.createElement("tr");
                row.innerHTML = `<td>${keyName}</td><td>${formattedValue}</td>`;
                statsTable.appendChild(row);
            }

            // Populate Historical Chart
            if (window.histChartObj) window.histChartObj.destroy();
            window.histChartObj = new Chart(histCtx, {
                type: "line",
                data: {
                    // Use REAL dates from the API
                    labels: data.historical_graph_data.dates, 
                    datasets: [{
                        label: `${data.ticker} (Past 30 Days)`,
                        // Use REAL prices from the API
                        data: data.historical_graph_data.prices, 
                        borderWidth: 2,
                        borderColor: "#00E0FF",
                        tension: 0.3
                    }]
                },
                options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { labels: { color: "white" } } }, scales: { x: { ticks: { color: "rgba(255,255,255,0.6)" } }, y: { ticks: { color: "rgba(255,255,255,0.6)" } } } }
            });

            // Populate Prediction Chart
            if (window.predChartObj) window.predChartObj.destroy();
            window.predChartObj = new Chart(predCtx, {
                type: "line",
                data: {
                    // Use REAL predicted dates from the API
                    labels: data.prediction_graph_data.dates, 
                    datasets: [{
                        label: `Predicted Price (₹)`,
                        // Use REAL predicted prices from the API
                        data: data.prediction_graph_data.prices, 
                        borderWidth: 2,
                        borderColor: "#14FFEC",
                        tension: 0.3,
                        fill: false
                    }]
                },
                options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { labels: { color: "white" } } }, scales: { x: { ticks: { color: "rgba(255,255,255,0.6)" } }, y: { ticks: { color: "rgba(255,255,255,0.6)" } } } }
            });

            // Populate Predicted Values Grid
            predGrid.innerHTML = ""; // Clear old grid
            data.prediction_graph_data.dates.forEach((date, i) => {
                const price = data.prediction_graph_data.prices[i];
                const div = document.createElement("div");
                div.className = "pred-item"; // From your old JS
                div.innerHTML = `<div class="pred-date">${date}</div><div class="pred-price">₹${price.toFixed(2)}</div>`;
                predGrid.appendChild(div);
            });

        } catch (error) {
            console.error("Failed to generate insights:", error);
            alert("Error: " + error.message);
            // Show placeholder again if it fails
            placeholderCard.style.display = "block";
            fullWidthCharts.style.display = "none";
        } finally {
            // 4. Reset button
            generateBtn.disabled = false;
            generateBtn.textContent = "Generate Insights";
        }
    }
});