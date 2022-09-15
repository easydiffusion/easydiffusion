export const CompletedImages = () => {
  return (
    <div className="completed-images">
      <h1>Completed Images</h1>
    </div>
  );
  // const { data } = useQuery("completedImages", getCompletedImages);
  // return (
  //   <div className="completed-images">
  //     <h2>Completed Images</h2>
  //     <div className="completed-images-list">
  //       {data?.map((image) => (
  //         <GeneratedImage imageData={image.data} key={image.id} />
  //       ))}
  //     </div>
  //   </div>
  // );
};
