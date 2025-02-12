// Initialize React components
const { useState, useEffect } = React;
const { 
    BarChart, Bar, XAxis, YAxis, 
    CartesianGrid, Tooltip, LineChart, Line 
} = Recharts;

// Card component for metrics
const MetricCard = ({ title, value, icon }) => (
    <div className="bg-white p-6 rounded-lg shadow-md">
        <div className="flex items-center justify-between">
            <h3 className="text-gray-500 text-sm font-medium">{title}</h3>
            <span className="text-gray-400">{icon}</span>
        </div>
        <p className="mt-2 text-3xl font-bold text-gray-900">{value}</p>
    </div>
);

// Main Dashboard Component
function Dashboard() {
    const [data, setData] = useState(null);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        const fetchData = async () => {
            try {
                // Initialize bridge connection
                await new Promise((resolve, reject) => {
                    if (window.bridge) {
                        resolve();
                    } else {
                        new QWebChannel(qt.webChannelTransport, function(channel) {
                            window.bridge = channel.objects.bridge;
                            resolve();
                        });
                    }
                });

                // Get data through the bridge
                const jsonStr = await window.bridge.readAnalyticsFile();
                console.log("Received data:", jsonStr.substring(0, 100)); // Log first 100 chars

                // Parse the JSON string
                let analyticsData;
                try {
                    console.log("Attempting to parse JSON:", jsonStr);
                    analyticsData = JSON.parse(jsonStr);
                    console.log("Parsed analytics data:", analyticsData);
                } catch (parseError) {
                    console.error("JSON Parse error:", parseError);
                    console.error("Raw JSON string:", jsonStr);
                    throw new Error(`Failed to parse JSON: ${parseError.message}`);
                }

                if (analyticsData.error) {
                    throw new Error(analyticsData.error);
                }

                // Process the data
                const processedData = processAnalyticsData(analyticsData);
                setData(processedData);
                setIsLoading(false);
            } catch (err) {
                console.error('Dashboard data loading error:', err);
                setError('Error loading dashboard data: ' + err.message);
                setIsLoading(false);
            }
        };

        fetchData();
    }, []);

    const processAnalyticsData = (analyticsData) => {
        console.log("Processing analytics data:", analyticsData);
        if (!analyticsData) {
            console.warn("No analytics data received");
            return {
                chartData: [],
                totalEngagements: 0,
                totalImpressions: 0,
                engagementRate: 0
            };
        }
        
        const engagement_stats = analyticsData.engagement_stats || {};
        const posts = analyticsData.posts || [];
        const summary = analyticsData.summary || {};
        
        if (!posts || !Array.isArray(posts)) {
            console.warn("No posts data available");
            return {
                chartData: [],
                totalEngagements: 0,
                totalImpressions: 0,
                engagementRate: 0
            };
        }

        const engagementByDay = {};
        
        // Process posts for chart data
        posts.forEach(post => {
            try {
                if (post && post.due_date) {
                    const date = new Date(post.due_date).toLocaleDateString();
                    const engagement = parseFloat(post.engagement_score) || 0;
                    
                    // Group engagements by date
                    engagementByDay[date] = (engagementByDay[date] || 0) + engagement;
                }
            } catch (error) {
                console.error("Error processing post:", post, error);
            }
        });

        // Convert engagement data for chart
        const chartData = Object.entries(engagementByDay)
            .map(([date, value]) => ({
                date,
                engagements: parseFloat(value.toFixed(2))
            }))
            .sort((a, b) => new Date(a.date) - new Date(b.date));

        console.log("Chart data:", chartData);
        console.log("Summary data:", summary);

        return {
            chartData,
            totalEngagements: (summary && summary.total_engagement) || 0,
            totalImpressions: (summary && summary.total_impressions) || 0,
            engagementRate: (summary && summary.engagement_rate) || 0
        };
    };

    // Render loading state
    if (isLoading) {
        return (
            <div className="flex items-center justify-center h-screen">
                <p className="text-lg text-gray-600">Loading dashboard data...</p>
            </div>
        );
    }

    // Render error state
    if (error) {
        return (
            <div className="flex items-center justify-center h-screen">
                <p className="text-lg text-red-600">{error}</p>
            </div>
        );
    }

    // Render no data state
    if (!data) {
        return (
            <div className="flex items-center justify-center h-screen">
                <p className="text-lg text-gray-600">No data available</p>
            </div>
        );
    }

    // Render dashboard
    return (
        <div className="p-6 max-w-7xl mx-auto">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
                <MetricCard 
                    title="Engagement Rate" 
                    value={`${Number(data.engagementRate).toFixed(1)}%`}
                    icon="📈"
                />
                <MetricCard 
                    title="Total Engagements" 
                    value={Math.round(data.totalEngagements)}
                    icon="👥"
                />
                <MetricCard 
                    title="Total Impressions" 
                    value={Math.round(data.totalImpressions)}
                    icon="👁️"
                />
            </div>

            <div className="bg-white p-6 rounded-lg shadow-md mb-6">
                <h2 className="text-xl font-bold mb-4">Engagement Over Time</h2>
                <div style={{ width: '100%', height: 400 }}>
                    <BarChart
                        width={800}
                        height={400}
                        data={data.chartData}
                        margin={{
                            top: 5,
                            right: 30,
                            left: 20,
                            bottom: 5,
                        }}
                    >
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis 
                            dataKey="date"
                            tick={{ fontSize: 12 }}
                        />
                        <YAxis />
                        <Tooltip />
                        <Bar 
                            dataKey="engagements" 
                            fill="#3b82f6"
                            name="Engagements"
                        />
                    </BarChart>
                </div>
            </div>
        </div>
    );
}

// Mount the React application
const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<Dashboard />);