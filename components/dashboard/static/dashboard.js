// post-pilot/components/dashboard/static/dashboard.js

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
const Dashboard = () => {
    const [data, setData] = useState(null);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        const fetchData = async () => {
            try {
                // Wait for QWebChannel to be initialized
                await new Promise(resolve => {
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
                const response = window.bridge.readAnalyticsFile("");
                const analyticsData = JSON.parse(response);
                
                if (analyticsData.error) {
                    throw new Error(analyticsData.error);
                }
                
                // Process the data
                const processedData = processAnalyticsData(analyticsData);
                setData(processedData);
                setIsLoading(false);
            } catch (err) {
                setError('Error loading dashboard data: ' + err.message);
                setIsLoading(false);
                console.error('Dashboard data loading error:', err);
            }
        };

        fetchData();
    }, []);

    const processAnalyticsData = (analyticsData) => {
        const { engagement_stats, posts } = analyticsData;
        
        if (!posts || !Array.isArray(posts)) {
            return {
                chartData: [],
                totalEngagements: 0,
                totalImpressions: 0,
                engagementRate: 0
            };
        }

        const engagementByDay = {};
        let totalEngagements = 0;
        let totalImpressions = 0;

        posts.forEach(post => {
            if (post && post.due_date) {
                const date = new Date(post.due_date).toLocaleDateString();
                engagementByDay[date] = (engagementByDay[date] || 0) + (post.engagement_score || 0);
                totalEngagements += post.engagement_score || 0;
                totalImpressions += post.impressions || 0;
            }
        });

        const chartData = Object.entries(engagementByDay)
            .map(([date, value]) => ({
                date,
                engagements: value
            }))
            .sort((a, b) => new Date(a.date) - new Date(b.date));

        return {
            chartData,
            totalEngagements,
            totalImpressions,
            engagementRate: (engagement_stats && engagement_stats.average) || 0
        };
    };

    if (isLoading) return (
        <div className="flex items-center justify-center h-screen">
            <p className="text-lg text-gray-600">Loading dashboard data...</p>
        </div>
    );

    if (error) return (
        <div className="flex items-center justify-center h-screen">
            <p className="text-lg text-red-600">{error}</p>
        </div>
    );

    if (!data) return null;

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
                    value={data.totalEngagements}
                    icon="👥"
                />
                <MetricCard 
                    title="Total Impressions" 
                    value={data.totalImpressions}
                    icon="👁️"
                />
            </div>

            <div className="bg-white p-6 rounded-lg shadow-md mb-6">
                <h2 className="text-xl font-bold mb-4">Engagement Over Time</h2>
                <div style={{ width: '100%', height: 400 }}>
                    <BarChart width={800} height={400} data={data.chartData}>
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
};

// Mount the React application
const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<Dashboard />);