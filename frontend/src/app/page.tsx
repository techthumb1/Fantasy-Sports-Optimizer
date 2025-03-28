import PlayerQuery from '../components/PlayerQuery';

export default function Home() {
  return (
    <>
      <section className="w-full max-w-5xl text-center mb-12">
        <h1 className="text-5xl font-semibold mb-4 bg-gradient-to-r from-black to-gray-900 p-4 rounded-lg">
          Fantasy Sports Optimizer
        </h1>
      </section>

      <section className="w-full max-w-5xl text-center mb-12">
        <h2 className="text-3xl font-semibold mb-4">
          Optimize Your Fantasy Sports Teams with AI
        </h2>
        <p className="text-lg">
          Get started by leveraging state-of-the-art models to enhance your football and basketball team performance.
        </p>
      </section>

      <section className="grid grid-cols-1 md:grid-cols-3 gap-8 w-full max-w-4xl mb-32 text-center mx-auto">
        <div className="group rounded-lg p-8 transition-colors bg-gray-800 hover:bg-gray-700">
          <h3 className="mb-3 text-2xl font-semibold">
            Documentation{" "}
            <span className="inline-block transition-transform group-hover:translate-x-1 motion-reduce:transform-none">
              -&gt;
            </span>
          </h3>
          <p className="m-0 max-w-[28ch] text-sm opacity-80">
            Find detailed information about the features and API of the Fantasy Sports Optimizer.
          </p>
        </div>

        <div className="group rounded-lg p-8 transition-colors bg-gray-800 hover:bg-gray-700">
          <h3 className="mb-3 text-2xl font-semibold">
            Learn{" "}
            <span className="inline-block transition-transform group-hover:translate-x-1 motion-reduce:transform-none">
              -&gt;
            </span>
          </h3>
          <p className="m-0 max-w-[28ch] text-sm opacity-80">
            Learn how to maximize your team&apos;s performance using our AI-powered tools.
          </p>
        </div>

        <div>
      <h1>Welcome to Fantasy Sports Optimizer</h1>
      <PlayerQuery />
         </div>

        <div className="group rounded-lg p-8 transition-colors bg-gray-800 hover:bg-gray-700">
          <h3 className="mb-3 text-2xl font-semibold">
            Models{" "}
            <span className="inline-block transition-transform group-hover:translate-x-1 motion-reduce:transform-none">
              -&gt;
            </span>
          </h3>
          <p className="m-0 max-w-[28ch] text-sm opacity-80">
            Explore and integrate cutting-edge models for precise predictions.
          </p>
        </div>
      </section>
    </>
  );
}
