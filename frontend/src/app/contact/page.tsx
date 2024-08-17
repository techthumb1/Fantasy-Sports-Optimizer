export default function Contact() {
    return (
      <div className="max-w-5xl mx-auto p-4 bg-gray-800 rounded-lg">
        <h1 className="text-4xl font-bold mb-16 text-center">Contact Us</h1>
        <div className="text-lg opacity-80 space-y-14 text-center">
          <p className="pl-4"> Email: support@fantasysportsoptimizer.com</p>
          <p className="pl-4"> Phone: 1-800-123-4567</p>
          <p className="pl-4"> Address: 123 Fantasy Way, Sportsville, USA</p>
        </div>
        <div className="mt-16 text-center">
          <a href="/" className="text-lg text-blue-500 hover:underline">Return to Home</a>
        </div>
      </div>
    );
  }